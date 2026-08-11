"""Task-level rollout termination metrics.

These flags are deliberately independent from ``mask_rollout``.  The latter
controls whether a trajectory contributes to the policy update; it is not a
description of why generation stopped.
"""


def classify_rollout_status(
    *,
    response_tokens: int,
    response_limit: int,
    main_context_tokens: int,
    working_context_limit: int,
    is_finish: bool,
    iteration: int,
    max_turn: int,
    timed_out: bool,
) -> dict[str, int | str]:
    """Classify how a root-agent rollout ended using mutually clear flags."""
    response_tokens = max(int(response_tokens), 0)
    response_limit = max(int(response_limit), 0)
    main_context_tokens = max(int(main_context_tokens), 0)
    working_context_limit = max(int(working_context_limit), 0)
    is_finish = bool(is_finish)

    # CallLLM stops issuing another generation when fewer than 10 response
    # tokens remain.  Treat that narrow tail as exhaustion of the configured
    # token budget, but never label a trajectory that emitted ``finish`` as
    # overlong.
    exhaustion_threshold = max(response_limit - 10, 0)
    hit_token_limit = bool(
        not is_finish
        and response_limit > 0
        and response_tokens >= exhaustion_threshold
    )
    hit_max_turn = bool(not is_finish and int(iteration) >= int(max_turn))
    hit_timeout = bool(not is_finish and timed_out)
    unfolded_main = bool(
        working_context_limit > 0
        and main_context_tokens > working_context_limit * 0.5
    )

    if is_finish:
        termination_reason = "finish"
    elif hit_timeout:
        termination_reason = "timeout"
    elif hit_token_limit:
        termination_reason = "token_limit"
    elif hit_max_turn:
        termination_reason = "max_turn"
    else:
        termination_reason = "stopped_without_finish"

    return {
        "overlong": int(hit_token_limit),
        "no_finish": int(not is_finish),
        "hit_token_limit": int(hit_token_limit),
        "hit_max_turn": int(hit_max_turn),
        "hit_timeout": int(hit_timeout),
        "unfolded_main": int(unfolded_main),
        "termination_reason": termination_reason,
    }
