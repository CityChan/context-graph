"""Opt-in ScienceWorld v3_long reward and stopping semantics.

Native evaluation remains binary. This profile is an ordinary GraphRPO
training ablation, not a change to the ScienceWorld benchmark score.
"""

from dataclasses import dataclass
import math


def validate_reward_profile(profile, *, ability, algorithm, is_train):
    if is_train:
        from agents.graph_rpo import reject_removed_graph_rpo_options
        reject_removed_graph_rpo_options(algorithm)
    if profile not in {"binary", "v3_long"}:
        raise ValueError(f"Unknown ScienceWorld training reward profile: {profile}")
    if profile == "binary":
        return
    if not str(ability).startswith("ScienceWorld@"):
        raise ValueError("v3_long requires a ScienceWorld task")
    if is_train and (
        str(algorithm.get("adv_estimator", "")).lower() not in {"graphrpo", "advantageestimator.graphrpo"}
        or algorithm.get("graphrpo_require_binary_reward", True)
    ):
        raise ValueError("v3_long requires ordinary GraphRPO with finite scalar rewards enabled")


def training_reward(*, completed, peak_score, natural_end, token_limit, no_tool_limit, env_failed=False):
    """Success first; partial progress only on natural termination.

    Token exhaustion and the consecutive no-tool cap receive -0.1. Other
    truncations (turn/time/environment-step caps) get zero, without partial
    credit. Infrastructure errors never earn progress or completion credit.
    """
    peak_score = float(peak_score)
    if not math.isfinite(peak_score):
        raise ValueError("ScienceWorld peak score must be finite")
    if env_failed:
        return 0.0
    if completed:
        return 1.0
    if natural_end:
        return 0.15 * min(max(peak_score, 0.0), 100.0) / 100.0
    return -0.1 if token_limit or no_tool_limit else 0.0


@dataclass
class NoToolGuard:
    """Episode-wide consecutive action turns without a parsed function call.

    Share between main and branch agents. Controller-only JSON turns do not
    enter this counter. Zero disables the guard for existing profiles.
    """

    limit: int = 0
    consecutive: int = 0
    stopped: bool = False

    def observe(self, has_tool):
        if self.stopped:
            return False
        self.consecutive = 0 if has_tool else self.consecutive + 1
        self.stopped = self.limit > 0 and self.consecutive >= self.limit
        return not self.stopped
