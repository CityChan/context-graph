"""Eval entry point for ScienceAgentBench (zero-shot).

Registers the three code-domain agent loops in verl's agent system:
  - react_agent_code              (vanilla ReAct, python_exec only)
  - fold_agent_code               (fold-style branch + return)
  - context_graph_code_isolated_agent (ContextGraph with v5 #1+#3)

The sbatch scripts select one via
  actor_rollout_ref.rollout.agent.default_agent_loop=<name>

Usage (eval; val_only=True):
  python -m scripts.train_sab algorithm.adv_estimator=foldgrpo \
      actor_rollout_ref.rollout.agent.default_agent_loop=react_agent_code \
      +actor_rollout_ref.rollout.plugin.workflow=code \
      data.val_files=data/sab_test.parquet \
      trainer.val_before_train=True trainer.val_only=True
"""

import logging
import os
from typing import Any, Union

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopOutput,
    register,
)
from verl import DataProto
from agents.react_agent_code import process_item as process_item_react
from agents.fold_agent_code import process_item as process_item_fold
from agents.graph_agent_code_isolated import process_item as process_item_graph
from agents.utils import CallLLM, TaskContext
from envs.scienceagent_sandbox import prewarm_heavy_imports

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _build_context(self, sampling_params, kwargs):
    """Shared TaskContext + LLM client setup for all three agent loops."""
    item = DataProto.from_dict(non_tensors=kwargs)
    llm_client = CallLLM(
        url=self.server_manager,
        tokenizer=self.tokenizer,
        config=self.config.actor_rollout_ref.rollout,
        loop=self.loop,
    )
    is_validate = kwargs.get('validate', False)
    context = TaskContext(
        config=self.config,
        global_step=kwargs.get('global_step', 0),
        llm_client=llm_client,
        is_train=not is_validate,
        tokenizer=self.tokenizer,
    )
    return item, context


@register("react_agent_code")
class ReactAgentCodeLoop(AgentLoopBase):
    """Single-thread ReAct over python_exec. Apples-to-apples baseline for
    Fold and ContextGraph on ScienceAgentBench."""

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
    """FoldAgent (branch + return) over python_exec."""

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
    """ContextGraph (branch + graph state + auto-bind + uniqueness) over
    python_exec. Inherits v5 Improvement #1 / #3 unchanged via context_graph."""

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


def main():
    """Entry point — runs verl's main PPO trainer with the three code
    agent loops registered above. Driven by hydra overrides from sbatch."""
    from verl.trainer.main_ppo import main as verl_main
    verl_main()


if __name__ == "__main__":
    main()
