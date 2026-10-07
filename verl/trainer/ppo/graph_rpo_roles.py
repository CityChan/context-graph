"""Deterministic shared-policy E/M schedule, indexed by saved rollout attempts."""


def graph_rpo_update_role(config, global_step):
    if not config.get("graphrpo_alternating_roles", False):
        return None
    if config.get("graphrpo_memory_only", False):
        raise ValueError("Alternating GraphRPO cannot also be memory-only")
    if global_step < 1:
        raise ValueError("Alternating GraphRPO requires a positive saved global step")
    return "E" if global_step % 2 else "M"
