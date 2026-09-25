"""Explicit evaluation modes; the legacy path remains reproducible."""

from copy import copy, deepcopy


def memory_mode(plugin):
    mode = getattr(plugin, "contextgraph_memory_mode", "legacy")
    if mode not in {"legacy", "repaired", "foldagent"}:
        raise ValueError(f"Unknown contextgraph_memory_mode: {mode}")
    return mode


def foldagent_inputs(item, context):
    """Delegate to the original executor with the original branch prompt.

    Copy request-local metadata/config, never the live LLM client or tokenizer.
    This is an equivalence control, not a second implementation of FoldAgent.
    """
    import numpy as np

    if context.is_train:
        raise ValueError("FoldAgent equivalence mode is evaluation-only")
    result = copy(item)
    result.non_tensor_batch = dict(item.non_tensor_batch)
    extra = np.asarray(item.non_tensor_batch["extra_info"], dtype=object)
    mapped = np.empty(extra.shape, dtype=object)
    for index in np.ndindex(extra.shape):
        metadata = deepcopy(extra[index])
        workflow = metadata.get("workflow") or context.config.actor_rollout_ref.rollout.plugin.workflow
        if workflow not in {"search_graph", "search_branch"}:
            raise ValueError("FoldAgent equivalence currently supports search workflows only")
        metadata["workflow"] = "search_branch"
        mapped[index] = metadata
    result.non_tensor_batch["extra_info"] = mapped
    local_context = copy(context)
    local_context.config = deepcopy(context.config)
    plugin = local_context.config.actor_rollout_ref.rollout.plugin
    plugin.workflow = "search_branch"
    plugin.process_reward = ["flat", "scope"]
    if hasattr(plugin, "controller_owned_tool_formatting"):
        plugin.controller_owned_tool_formatting = False
    return result, local_context


async def run_foldagent_equivalent(item, context):
    from .fold_agent import process_item

    item, context = foldagent_inputs(item, context)
    return await process_item(item, context)
