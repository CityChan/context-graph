"""Training entry point for vanilla ReAct GRPO baseline.

No graph operations, no branching, no fold-summarize, no forced consolidation.
Just linear chat history with the BrowseComp-Plus search tool, trained with
GRPO end-to-end. Step 0 of this trainer (val_before_train=True) is the
true zero-shot ReAct baseline.

Usage:
    python -m scripts.train_baseline algorithm.adv_estimator=foldgrpo \
        actor_rollout_ref.rollout.agent.default_agent_loop=react_agent \
        +actor_rollout_ref.rollout.plugin.workflow=search_base \
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
from agents.react_agent import process_item
from agents.utils import CallLLM, TaskContext

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("react_agent")
class ReactAgentLoop(AgentLoopBase):
    """Vanilla ReAct: linear chat history + search tool. No branch, no graph,
    no consolidation. Apples-to-apples baseline for FoldAgent and ContextGraph
    on the same BC-Plus train/test split."""

    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True

        logger.info("Initializing ReactAgentLoop class")

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
        rollout_results = await process_item(item, context)

        return rollout_results


def main():
    """Entry point for training - runs VERL's main PPO trainer with react_agent registered."""
    from verl.trainer.main_ppo import main as verl_main
    verl_main()


if __name__ == "__main__":
    main()
