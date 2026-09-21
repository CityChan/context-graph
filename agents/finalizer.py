"""Shared emergency final-answer policy for finite-context agent rollouts."""

import re


OBSERVATION_TRUNCATION_MARKER = (
    "\n...[observation truncated to preserve the final-answer budget]...\n"
)


FINAL_ANSWER_PROMPT = """FINAL ANSWER REQUIRED NOW.
Stop researching and do not call search, open_page, branch, or return. Using only the evidence already present in the conversation, submit your single best answer immediately with the finish tool. A best-effort answer is required even if some evidence is incomplete. Use exactly:
<function=finish>
<parameter=answer>YOUR BEST ANSWER</parameter>
<parameter=explanation>Brief evidence-based explanation with citations already found, if available.</parameter>
<parameter=confidence>YOUR CONFIDENCE</parameter>
</function>"""


def remaining_generation_tokens(agent) -> int:
    """Return tokens left in the rollout's prompt+response budget."""
    budget_end = int(agent.prompt_ids_len) + int(agent.config.response_length)
    return max(budget_end - len(agent.context()), 0)


def _truncate_observation_tokens(agent, observation: str, keep_tokens: int) -> str:
    """Keep the head and tail of an observation within a token target."""
    observation = str(observation)
    token_ids = agent.tokenizer.encode(observation, add_special_tokens=False)
    keep_tokens = max(int(keep_tokens), 0)
    if len(token_ids) <= keep_tokens:
        return observation
    if keep_tokens == 0:
        return ""

    marker_ids = agent.tokenizer.encode(
        OBSERVATION_TRUNCATION_MARKER, add_special_tokens=False
    )
    if keep_tokens <= len(marker_ids) + 2:
        return agent.tokenizer.decode(token_ids[:keep_tokens], skip_special_tokens=True)

    payload_tokens = keep_tokens - len(marker_ids)
    head_tokens = (payload_tokens + 1) // 2
    tail_tokens = payload_tokens - head_tokens
    head = agent.tokenizer.decode(
        token_ids[:head_tokens], skip_special_tokens=True
    )
    tail = agent.tokenizer.decode(
        token_ids[-tail_tokens:], skip_special_tokens=True
    ) if tail_tokens else ""
    return f"{head}{OBSERVATION_TRUNCATION_MARKER}{tail}"


def append_observation_preserving_final_answer(
    agent,
    observation: str,
    reserve_tokens: int,
    safety_tokens: int = 64,
):
    """Append the largest observation that cannot consume protected budget.

    Returns the text actually appended, or ``None`` when even the smallest
    user turn would cross the protected boundary. The check measures the
    rendered context, so chat-template overhead is included.
    """
    observation = str(observation)
    reserve_tokens = max(int(reserve_tokens or 0), 0)
    safety_tokens = max(int(safety_tokens or 0), 0)
    if reserve_tokens == 0:
        agent.append({"role": "user", "content": observation})
        return observation

    protected_tokens = reserve_tokens + safety_tokens
    context_len_before = len(agent.context())
    max_append_tokens = remaining_generation_tokens(agent) - protected_tokens
    if max_append_tokens <= 0:
        print(
            "[FINALIZER] skipping observation: only protected final-answer "
            f"budget remains ({remaining_generation_tokens(agent)} tokens)"
        )
        return None

    def fits(candidate: str) -> bool:
        agent.append({"role": "user", "content": candidate})
        rendered_cost = len(agent.context()) - context_len_before
        enough_room = remaining_generation_tokens(agent) >= protected_tokens
        agent.rollback(k=1)
        return rendered_cost <= max_append_tokens and enough_room

    if fits(observation):
        agent.append({"role": "user", "content": observation})
        return observation

    token_ids = agent.tokenizer.encode(observation, add_special_tokens=False)
    low, high = 1, max(len(token_ids) - 1, 0)
    fitted = None
    while low <= high:
        mid = (low + high) // 2
        candidate = _truncate_observation_tokens(agent, observation, mid)
        if fits(candidate):
            fitted = candidate
            low = mid + 1
        else:
            high = mid - 1

    if fitted is None:
        print(
            "[FINALIZER] skipping observation: chat-template overhead would "
            "consume the protected final-answer budget"
        )
        return None

    agent.append({"role": "user", "content": fitted})
    print(
        "[FINALIZER] truncated observation to preserve "
        f"{protected_tokens} protected tokens"
    )
    return fitted


async def step_preserving_final_answer(
    agent,
    reserve_tokens: int,
    *,
    completion_kwargs=None,
    context_overlay: str | None = None,
):
    """Generate one normal turn with an optional transient context overlay.

    The overlay is visible to this generation only. Restoring the original
    user turn prevents repeatedly injected memory snapshots from accumulating
    in the persistent trajectory.
    """
    overlay_idx = None
    original_content = None
    context_overlay = str(context_overlay or "").strip()
    if context_overlay and agent.messages() and agent.messages()[-1].get("role") == "user":
        overlay_idx = len(agent.messages()) - 1
        original_content = agent.messages()[overlay_idx].get("content", "")
        agent.replace_user_turn(
            overlay_idx,
            f"{original_content}\n\n{context_overlay}",
        )

    try:
        reserve_tokens = max(int(reserve_tokens or 0), 0)
        if reserve_tokens == 0:
            if completion_kwargs is None:
                return await agent.step()
            return await agent.step(completion_kwargs=completion_kwargs)

        normal_budget = remaining_generation_tokens(agent) - reserve_tokens
        if normal_budget < 10:
            return None
        if completion_kwargs is None:
            return await agent.step(max_new_tokens=normal_budget)
        return await agent.step(
            max_new_tokens=normal_budget, completion_kwargs=completion_kwargs
        )
    finally:
        if overlay_idx is not None:
            agent.replace_user_turn(overlay_idx, original_content)


def _fallback_finish_call(response: str) -> str:
    """Wrap non-tool finalizer output in a parseable best-effort finish call."""
    answer = (response or "Best effort: unable to determine a more specific answer.").strip()
    answer = re.sub(r"</?function(?:=[^>]+)?>", " ", answer)
    answer = re.sub(r"</?parameter(?:=[^>]+)?>", " ", answer)
    answer = " ".join(answer.split())
    if not answer:
        answer = "Best effort: unable to determine a more specific answer."
    return (
        "<function=finish>\n"
        f"<parameter=answer>{answer}</parameter>\n"
        "<parameter=explanation>Best-effort answer from the evidence already gathered.</parameter>\n"
        "<parameter=confidence>low</parameter>\n"
        "</function>"
    )


async def submit_emergency_final_answer(
    agent,
    env,
    reserve_tokens: int,
    action_runner,
    *,
    context_overlay: str | None = None,
) -> bool:
    """Spend the protected reserve on one final answer and submit it to the env.

    The model gets the complete current trajectory. If it ignores the requested
    finish schema, its generated best effort is wrapped in a finish call so an
    exhausted rollout still submits an answer instead of silently terminating.
    """
    reserve_tokens = max(int(reserve_tokens or 0), 0)
    if reserve_tokens == 0:
        return False
    if getattr(env, "is_finish", False) or getattr(env, "finish", False):
        return False

    overlay_idx = None
    persistent_content = None
    context_overlay = str(context_overlay or "").strip()
    if agent.messages() and agent.messages()[-1].get("role") == "user":
        idx = len(agent.messages()) - 1
        content = agent.messages()[idx].get("content", "")
        persistent_content = f"{content}\n\n{FINAL_ANSWER_PROMPT}"
        generation_content = persistent_content
        if context_overlay:
            # The latest raw observation has already been integrated into the
            # memory graph. Replace it for this final call to reclaim context,
            # then restore an auditable prompt-only trajectory afterward.
            generation_content = f"{context_overlay}\n\n{FINAL_ANSWER_PROMPT}"
        agent.replace_user_turn(idx, generation_content)
        overlay_idx = idx
    else:
        persistent_content = FINAL_ANSWER_PROMPT
        generation_content = (
            f"{context_overlay}\n\n{FINAL_ANSWER_PROMPT}"
            if context_overlay else FINAL_ANSWER_PROMPT
        )
        agent.append({"role": "user", "content": generation_content})
        overlay_idx = len(agent.messages()) - 1

    final_budget = min(reserve_tokens, remaining_generation_tokens(agent))
    if final_budget < 10:
        print(f"[FINALIZER] insufficient protected budget: {final_budget}")
        if context_overlay and overlay_idx is not None:
            agent.replace_user_turn(overlay_idx, persistent_content)
        return False

    print(f"[FINALIZER] forcing best-effort answer with {final_budget} tokens")
    try:
        response = await agent.step(max_new_tokens=final_budget)
    finally:
        if context_overlay and overlay_idx is not None:
            agent.replace_user_turn(overlay_idx, persistent_content)
    if response is None:
        return False

    valid_finish = bool(
        "<function=finish>" in response
        and re.search(r"<parameter=answer>\s*\S.*?</parameter>", response, re.DOTALL)
    )
    if hasattr(env, "emergency_finish_wrapped"):
        env.emergency_finish_wrapped = not valid_finish
    candidate = response if valid_finish else _fallback_finish_call(response)
    # Some environments reject the first finish only to clear a one-time guard
    # (must-search / do-not-give-up). Re-submit the identical answer once.
    for _ in range(2):
        observation = await action_runner(env, candidate)
        if observation is None:
            return True
    return False
