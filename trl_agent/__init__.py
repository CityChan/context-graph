"""TRL-backed training support for FoldAgent and ContextGraph."""

from .rollout import ColocatedVLLMBatcher, RolloutResult

__all__ = ["ColocatedVLLMBatcher", "RolloutResult"]
