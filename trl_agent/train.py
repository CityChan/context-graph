"""Command-line entry point for TRL-backed FoldAgent/ContextGraph training."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from datasets import Dataset, load_dataset
from omegaconf import OmegaConf
from trl import TrlParser

from utils.configs import ModelConfig, QeRLConfig

from scripts.prepare_gsm8k_grpo_data import convert_example
from trl_agent.modeling import build_model_and_peft
from trl_agent.trainer import AgentGRPOTrainer


@dataclass
class AgentTrainingConfig:
    agent_kind: str = field(
        default="foldagent",
        metadata={"help": "Agent implementation: foldagent or contextgraph"},
    )
    agent_config: str = field(
        default="recipes/trl_agent/foldagent_gsm8k.yaml",
        metadata={"help": "OmegaConf file defining rollout plugin and algorithm settings"},
    )
    train_data_path: str | None = field(
        default=None,
        metadata={"help": "Optional local Parquet file; defaults to openai/gsm8k"},
    )
    train_max_samples: int = field(default=0)
    batch_wait_ms: float = field(default=2.0)


def load_agent_dataset(config: AgentTrainingConfig) -> Dataset:
    if config.train_data_path:
        dataset = load_dataset(
            "parquet", data_files={"train": config.train_data_path}, split="train"
        )
    else:
        source = load_dataset("openai/gsm8k", "main", split="train")
        rows = [
            convert_example(example, "train", index)
            for index, example in enumerate(source)
        ]
        dataset = Dataset.from_list(rows)
    if config.train_max_samples > 0:
        dataset = dataset.select(range(min(config.train_max_samples, len(dataset))))
    return dataset


def agent_reward_placeholder(*, completions, **kwargs):
    """Parent-trainer placeholder; rewards are supplied by the agent environment."""
    del kwargs
    return [0.0] * len(completions)


def main(
    agent_args: AgentTrainingConfig,
    training_args: QeRLConfig,
    model_args: ModelConfig,
) -> None:
    config_path = Path(agent_args.agent_config).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"agent config does not exist: {config_path}")
    agent_config = OmegaConf.load(config_path)
    agent_config.actor_rollout_ref.rollout.prompt_length = training_args.max_prompt_length
    agent_config.actor_rollout_ref.rollout.response_length = training_args.max_completion_length

    model, tokenizer, peft_config, noise_scheduler = build_model_and_peft(
        model_args, training_args
    )
    dataset = load_agent_dataset(agent_args)
    trainer = AgentGRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[agent_reward_placeholder],
        peft_config=peft_config,
        args=training_args,
        train_dataset=dataset,
        sigma_start=model_args.sigma_start,
        sigma_end=model_args.sigma_end,
        num_stages=model_args.num_stages,
        noise_scheduler=noise_scheduler,
        agent_config=agent_config,
        agent_kind=agent_args.agent_kind,
        batch_wait_ms=agent_args.batch_wait_ms,
    )
    has_checkpoint = (
        os.path.isdir(training_args.output_dir)
        and any(name.startswith("checkpoint-") for name in os.listdir(training_args.output_dir))
    )
    trainer.train(resume_from_checkpoint=True if has_checkpoint else None)


if __name__ == "__main__":
    parser = TrlParser((AgentTrainingConfig, QeRLConfig, ModelConfig))
    parsed_agent_args, parsed_training_args, parsed_model_args = parser.parse_args_and_config()
    main(parsed_agent_args, parsed_training_args, parsed_model_args)
