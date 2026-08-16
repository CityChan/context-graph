"""Shared emergency final-answer policy for finite-context agent rollouts."""

import re


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


async def step_preserving_final_answer(agent, reserve_tokens: int):
    """Generate one normal turn without consuming the final-answer reserve."""
    reserve_tokens = max(int(reserve_tokens or 0), 0)
    if reserve_tokens == 0:
        return await agent.step()

    normal_budget = remaining_generation_tokens(agent) - reserve_tokens
    if normal_budget < 10:
        return None
    return await agent.step(max_new_tokens=normal_budget)


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
    agent, env, reserve_tokens: int, action_runner
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

    if agent.messages() and agent.messages()[-1].get("role") == "user":
        idx = len(agent.messages()) - 1
        content = agent.messages()[idx].get("content", "")
        agent.replace_user_turn(idx, f"{content}\n\n{FINAL_ANSWER_PROMPT}")
    else:
        agent.append({"role": "user", "content": FINAL_ANSWER_PROMPT})

    final_budget = min(reserve_tokens, remaining_generation_tokens(agent))
    if final_budget < 10:
        print(f"[FINALIZER] insufficient protected budget: {final_budget}")
        return False

    print(f"[FINALIZER] forcing best-effort answer with {final_budget} tokens")
    response = await agent.step(max_new_tokens=final_budget)
    if response is None:
        return False

    valid_finish = bool(
        "<function=finish>" in response
        and re.search(r"<parameter=answer>\s*\S.*?</parameter>", response, re.DOTALL)
    )
    candidate = response if valid_finish else _fallback_finish_call(response)
    # Some environments reject the first finish only to clear a one-time guard
    # (must-search / do-not-give-up). Re-submit the identical answer once.
    for _ in range(2):
        observation = await action_runner(env, candidate)
        if observation is None:
            return True
    return False
