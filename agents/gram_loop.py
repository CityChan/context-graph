"""Hydra-loaded VERL adapter; imported only by training workers."""
import os
from pathlib import Path
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase
from agents.utils import CallLLM, TaskContext
from scripts.eval_gram import jsonl_writer
from scripts.train_gram import process_item


class GramAgentLoop(AgentLoopBase):
    async def run(self, sampling_params, **kwargs):
        client = CallLLM(url=self.server_manager, tokenizer=self.tokenizer,
                         config=self.config.actor_rollout_ref.rollout,
                         loop=self.loop, sampling_params=sampling_params)
        context = TaskContext(config=self.config, global_step=kwargs.get("global_step", 0),
                              llm_client=client, is_train=not kwargs.get("validate", False), tokenizer=self.tokenizer)
        directory = Path(os.environ.get("GRAM_TRACE_DIR", "outputs/gram-training-traces"))
        directory.mkdir(parents=True, exist_ok=True)
        return await process_item(kwargs, context, audit=jsonl_writer(directory / (uuid4().hex+".jsonl")))
