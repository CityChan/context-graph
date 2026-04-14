"""Training entry point for ContextGraph agent loop.

Registers the context_graph_agent in verl's agent loop system and
provides a main() entry point for VERL PPO training.

Usage:
    python -m scripts.train_graph algorithm.adv_estimator=foldgrpo \
        actor_rollout_ref.rollout.agent.default_agent_loop=context_graph_agent \
        +actor_rollout_ref.rollout.plugin.workflow=search_graph \
        ...
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
from agents.graph_agent import process_item as process_item_global
from agents.graph_agent_isolated import process_item as process_item_isolated
from agents.utils import CallLLM, TaskContext

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("context_graph_agent")
class ContextGraphAgentLoop(AgentLoopBase):
    """Original ContextGraph variant: single global graph, every observation
    injects the full graph state back into the main agent's prompt.
    Quadratic main-agent context growth → OOM-prone on long horizons."""

    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True

        logger.info("Initializing ContextGraphAgentLoop class")

        cls.tokenizer = tokenizer
        cls.processor = processor
        cls.config = config

    async def run(
        self, sampling_params: dict[str, Any], **kwargs
    ) -> Union[AgentLoopOutput, list[AgentLoopOutput]]:
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
        rollout_results = await process_item_global(item, context)

        return rollout_results


@register("context_graph_isolated_agent")
class ContextGraphIsolatedAgentLoop(AgentLoopBase):
    """Hierarchical/isolated ContextGraph variant: each branch runs on its
    own private subgraph; the main agent only ever sees parent-graph state
    (subtask + summary nodes). Main-agent context growth is linear, on par
    with FoldAgent, while preserving cross-branch graph reasoning at the
    parent level."""

    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True

        logger.info("Initializing ContextGraphIsolatedAgentLoop class")

        cls.tokenizer = tokenizer
        cls.processor = processor
        cls.config = config

    async def run(
        self, sampling_params: dict[str, Any], **kwargs
    ) -> Union[AgentLoopOutput, list[AgentLoopOutput]]:
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
        rollout_results = await process_item_isolated(item, context)

        return rollout_results


def main():
    """Entry point for training - runs VERL's main PPO trainer with context_graph_agent registered."""
    from verl.trainer.main_ppo import main as verl_main
    verl_main()


if __name__ == "__main__":
    main()
