"""Code-domain agent-loop registrations shared by SAB and DiscoveryBench.

This module intentionally lives beside the VERL registry implementation rather
than in an executable ``scripts.train_*`` module. Ray workers import the
``verl.experimental.agent_loop`` package to populate their local registry, so
importing an entry point from that package creates a circular dependency when
the entry point itself imports the registry.
"""

import logging
import os
from typing import Any

from verl import DataProto

from agents.fold_agent_code import process_item as process_item_fold
from agents.graph_agent_code_isolated import process_item as process_item_graph
from agents.react_agent_code import process_item as process_item_react
from agents.utils import CallLLM, TaskContext
from envs.scienceagent_sandbox import prewarm_heavy_imports

from .agent_loop import AgentLoopBase, register

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _build_context(self, sampling_params, kwargs):
    """Shared TaskContext and LLM client setup for code-domain loops."""
    item = DataProto.from_dict(non_tensors=kwargs)
    llm_client = CallLLM(
        url=self.server_manager,
        tokenizer=self.tokenizer,
        config=self.config.actor_rollout_ref.rollout,
        loop=self.loop,
        sampling_params=sampling_params,
    )
    is_validate = kwargs.get("validate", False)
    context = TaskContext(
        config=self.config,
        global_step=kwargs.get("global_step", 0),
        llm_client=llm_client,
        is_train=not is_validate,
        tokenizer=self.tokenizer,
    )
    return item, context


@register("react_agent_code")
class ReactAgentCodeLoop(AgentLoopBase):
    """Single-thread ReAct over python_exec."""

    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        logger.info("Initializing ReactAgentCodeLoop class")
        cls.tokenizer = tokenizer
        cls.processor = processor
        cls.config = config
        prewarm_heavy_imports()

    async def run(self, sampling_params: dict[str, Any], **kwargs):
        item, context = _build_context(self, sampling_params, kwargs)
        return await process_item_react(item, context)


@register("fold_agent_code")
class FoldAgentCodeLoop(AgentLoopBase):
    """FoldAgent branch/return loop over python_exec."""

    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        logger.info("Initializing FoldAgentCodeLoop class")
        cls.tokenizer = tokenizer
        cls.processor = processor
        cls.config = config
        prewarm_heavy_imports()

    async def run(self, sampling_params: dict[str, Any], **kwargs):
        item, context = _build_context(self, sampling_params, kwargs)
        return await process_item_fold(item, context)


@register("context_graph_code_isolated_agent")
class ContextGraphCodeIsolatedLoop(AgentLoopBase):
    """Isolated ContextGraph loop over python_exec."""

    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        logger.info("Initializing ContextGraphCodeIsolatedLoop class")
        cls.tokenizer = tokenizer
        cls.processor = processor
        cls.config = config
        prewarm_heavy_imports()

    async def run(self, sampling_params: dict[str, Any], **kwargs):
        item, context = _build_context(self, sampling_params, kwargs)
        return await process_item_graph(item, context)


__all__ = [
    "ReactAgentCodeLoop",
    "FoldAgentCodeLoop",
    "ContextGraphCodeIsolatedLoop",
]
