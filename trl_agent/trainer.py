"""QeRL/TRL trainer integration for FoldAgent and ContextGraph.

This module intentionally subclasses QeRL's fast colocated-vLLM trainer.  The
agent loop remains the source of trajectory construction; this layer only
adapts its variable multi-stream output into TRL tensors and applies the
project's FoldGRPO or GraphRPO objective.
"""

from __future__ import annotations

import asyncio
import copy
import time
from collections import defaultdict
from typing import Any
from uuid import uuid4

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from agents.graph_rpo import ANSWER_LIKELIHOOD_BACKENDS, graph_rpo_credit_backend
from agents.utils import CallLLM, TaskContext
from trl_trainer.grpo_trainer import GRPOTrainer
from verl import DataProto
from verl.trainer.ppo.core_algos import (
    compute_foldgrpo_advantage,
    compute_graphrpo_advantage,
    compute_graphrpo_loss_weights,
)

from .rollout import ColocatedVLLMBatcher


def _python_scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.item() if value.size == 1 else tuple(value.tolist())
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def _pad_rows(
    rows: list[list[Any]],
    *,
    length: int,
    value: float | int,
    dtype: torch.dtype,
    device: torch.device,
    left: bool = False,
) -> torch.Tensor:
    result = torch.full((len(rows), length), value, dtype=dtype, device=device)
    for index, row in enumerate(rows):
        kept = list(row[-length:] if left else row[:length])
        if not kept:
            continue
        tensor = torch.tensor(kept, dtype=dtype, device=device)
        if left:
            result[index, length - len(kept) :] = tensor
        else:
            result[index, : len(kept)] = tensor
    return result


class AgentGRPOTrainer(GRPOTrainer):
    """Fast single-GPU TRL trainer for project-native agent trajectories."""

    def __init__(
        self,
        *args: Any,
        agent_config: DictConfig | dict[str, Any],
        agent_kind: str,
        batch_wait_ms: float = 2.0,
        **kwargs: Any,
    ) -> None:
        self.agent_config = (
            agent_config
            if isinstance(agent_config, DictConfig)
            else OmegaConf.create(agent_config)
        )
        self.agent_kind = str(agent_kind).strip().lower()
        self.agent_batch_wait_ms = float(batch_wait_ms)
        if self.agent_kind == "foldagent":
            from agents.fold_agent import process_item
        elif self.agent_kind == "contextgraph":
            from agents.graph_agent_isolated import process_item
        else:
            raise ValueError("agent_kind must be 'foldagent' or 'contextgraph'")
        self.agent_process_item = process_item
        super().__init__(*args, **kwargs)
        self._validate_agent_training_config()

    @property
    def advantage_estimator(self) -> str:
        return str(self.agent_config.algorithm.adv_estimator).strip().lower()

    def _validate_agent_training_config(self) -> None:
        if not self.use_vllm or self.vllm_mode != "colocate":
            raise ValueError("AgentGRPOTrainer requires colocated vLLM")
        if self.accelerator.num_processes != 1:
            raise NotImplementedError(
                "The initial TRL agent backend is single-GPU; use verl for distributed runs"
            )
        if self.beta != 0.0:
            raise NotImplementedError("TRL agent training currently requires beta=0")
        if self.use_liger_loss:
            raise NotImplementedError("TRL agent training does not support the fused Liger loss")
        if self.num_generations < 2:
            raise ValueError("FoldGRPO/GraphRPO require num_generations >= 2")
        expected = "foldgrpo" if self.agent_kind == "foldagent" else "graphrpo"
        if self.advantage_estimator != expected:
            raise ValueError(
                f"{self.agent_kind} requires algorithm.adv_estimator={expected}"
            )
        if self.advantage_estimator == "graphrpo":
            backend = graph_rpo_credit_backend(
                self.agent_config.actor_rollout_ref.rollout.plugin
            )
            if backend in ANSWER_LIKELIHOOD_BACKENDS:
                raise NotImplementedError(
                    f"GraphRPO backend {backend!r} needs trainer-side reference-answer "
                    "scoring and is not yet supported by the TRL backend; use "
                    "old_policy_counterfactual_qa or external_evaluator"
                )

    def _make_agent_item(self, example: dict[str, Any], position: int) -> DataProto:
        values = copy.deepcopy(example)
        extra_info = values.get("extra_info") or {}
        question_uid = values.get("uid")
        if question_uid is None:
            question_uid = extra_info.get("index", extra_info.get("query"))
        if question_uid is None:
            question_uid = repr(values.get("prompt", position))
        values["uid"] = str(question_uid)
        values["gen_uid"] = f"{question_uid}:{self.state.global_step}:{position}:{uuid4().hex[:8]}"
        values["global_step"] = int(self.state.global_step)
        values["validate"] = not self.model.training
        return DataProto.from_dict(non_tensors=values)

    async def _run_agent_batch(self, inputs: list[dict[str, Any]], batcher: ColocatedVLLMBatcher):
        sampling_params = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": -1 if self.top_k is None else self.top_k,
            "min_p": 0.0 if self.min_p is None else self.min_p,
            "repetition_penalty": self.repetition_penalty,
            "logprobs": 1,
        }
        loop = asyncio.get_running_loop()
        client = CallLLM(
            url=batcher,
            tokenizer=self.processing_class,
            config=self.agent_config.actor_rollout_ref.rollout,
            loop=loop,
            sampling_params=sampling_params,
        )
        context = TaskContext(
            config=self.agent_config,
            global_step=int(self.state.global_step),
            is_train=self.model.training,
            tokenizer=self.processing_class,
            llm_client=client,
        )
        tasks = [
            self.agent_process_item(self._make_agent_item(example, index), context)
            for index, example in enumerate(inputs)
        ]
        nested = await asyncio.gather(*tasks)
        outputs = []
        for value in nested:
            outputs.extend(value if isinstance(value, list) else [value])
        return outputs

    def _generate_and_score_completions(self, inputs: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"
        if self.args.vllm_enable_sleep_mode:
            torch.cuda.empty_cache()
            self.llm.wake_up()
        if self.state.global_step != self._last_loaded_step:
            if self.unmerged_lora:
                self._move_peft_model_to_vllm()
            else:
                self._move_model_to_vllm()
            self._last_loaded_step = self.state.global_step

        batcher = ColocatedVLLMBatcher(self.llm, batch_wait_ms=self.agent_batch_wait_ms)
        started = time.perf_counter()
        outputs = asyncio.run(self._run_agent_batch(inputs, batcher))
        rollout_seconds = time.perf_counter() - started
        if self.args.vllm_enable_sleep_mode:
            self.llm.sleep(level=1)
        outputs = [output for output in outputs if output.response_ids and any(output.response_mask)]
        if not outputs:
            raise RuntimeError("agent batch produced no trajectory with policy tokens")

        prompt_length = int(self.agent_config.actor_rollout_ref.rollout.prompt_length)
        response_length = int(self.agent_config.actor_rollout_ref.rollout.response_length)
        prompt_rows = [output.prompt_ids for output in outputs]
        response_rows = [output.response_ids for output in outputs]
        prompt_ids = _pad_rows(
            prompt_rows,
            length=prompt_length,
            value=self.pad_token_id,
            dtype=torch.long,
            device=device,
            left=True,
        )
        completion_ids = _pad_rows(
            response_rows,
            length=response_length,
            value=self.pad_token_id,
            dtype=torch.long,
            device=device,
        )
        prompt_mask = _pad_rows(
            [[1] * min(len(row), prompt_length) for row in prompt_rows],
            length=prompt_length,
            value=0,
            dtype=torch.long,
            device=device,
            left=True,
        )
        completion_attention_mask = _pad_rows(
            [[1] * min(len(row), response_length) for row in response_rows],
            length=response_length,
            value=0,
            dtype=torch.long,
            device=device,
        )
        policy_mask = _pad_rows(
            [output.response_mask for output in outputs],
            length=response_length,
            value=0,
            dtype=torch.float32,
            device=device,
        ) * completion_attention_mask
        optimization_mask = torch.tensor(
            [not bool(output.extra_fields.get("mask_rollout", False)) for output in outputs],
            dtype=torch.float32,
            device=device,
        ).unsqueeze(1)
        completion_mask = policy_mask * optimization_mask
        rollout_logps = _pad_rows(
            [output.response_logprobs or [] for output in outputs],
            length=response_length,
            value=0.0,
            dtype=torch.float32,
            device=device,
        )
        process_reward_mask = _pad_rows(
            [output.extra_fields.get("process_reward_mask", []) for output in outputs],
            length=response_length,
            value=0.0,
            dtype=torch.float32,
            device=device,
        ) * policy_mask
        graph_edit_credit_mask = _pad_rows(
            [output.extra_fields.get("graph_edit_credit_mask", []) for output in outputs],
            length=response_length,
            value=0.0,
            dtype=torch.float32,
            device=device,
        ) * policy_mask

        rewards = torch.tensor(
            [float(output.reward_score or 0.0) for output in outputs],
            dtype=torch.float32,
            device=device,
        )
        token_rewards = torch.zeros_like(policy_mask)
        response_sizes = completion_attention_mask.sum(dim=1).long().clamp(min=1)
        token_rewards[torch.arange(len(outputs), device=device), response_sizes - 1] = rewards
        question_ids = np.array(
            [_python_scalar(output.extra_fields.get("uid")) for output in outputs],
            dtype=object,
        )
        episode_ids = np.array(
            [_python_scalar(output.extra_fields.get("gen_uid")) for output in outputs],
            dtype=object,
        )

        algo_config = self.agent_config.algorithm
        if self.advantage_estimator == "foldgrpo":
            advantages, _ = compute_foldgrpo_advantage(
                token_level_rewards=token_rewards,
                response_mask=policy_mask,
                index=question_ids,
                gen_uid=episode_ids,
                epsilon=float(algo_config.get("epsilon", 1e-6)),
                norm_adv_by_std_in_grpo=bool(algo_config.get("norm_adv_by_std_in_grpo", True)),
                fix_bad_positive_adv=bool(algo_config.get("fix_bad_positive_adv", False)),
                process_reward_mask=process_reward_mask,
                raw_token_level_scores=token_rewards,
                config=algo_config,
            )
            graph_loss_weights = None
            fold_loss_weights = torch.full(
                (len(outputs),),
                1.0 / len(outputs),
                dtype=torch.float32,
                device=device,
            )
        else:
            advantages, _ = compute_graphrpo_advantage(
                token_level_rewards=token_rewards,
                response_mask=policy_mask,
                index=question_ids,
                gen_uid=episode_ids,
                epsilon=float(algo_config.get("graphrpo_epsilon", 1e-6)),
                process_reward_mask=process_reward_mask,
                graph_edit_credit_mask=graph_edit_credit_mask,
                config=algo_config,
            )
            graph_loss_weights = compute_graphrpo_loss_weights(
                response_mask=policy_mask,
                index=question_ids,
                gen_uid=episode_ids,
            ) * optimization_mask
            fold_loss_weights = None

        # Match QeRL/TRL's on-policy semantics. When generation boundaries are
        # aligned with optimizer boundaries, the old policy is the current
        # training model and can be represented by detached current log-probs.
        # vLLM log-probs remain telemetry only because small backend numerical
        # differences must not create artificial PPO clipping.
        generate_every = self.args.steps_per_generation * self.num_iterations
        old_per_token_logps = None
        old_policy_recomputed = self.args.gradient_accumulation_steps % generate_every != 0
        if old_policy_recomputed:
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            attention_mask = torch.cat([prompt_mask, completion_attention_mask], dim=1)
            with torch.no_grad():
                old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    completion_ids.size(1),
                    batch_size=self.args.per_device_train_batch_size,
                )
        self._metrics[mode]["training/old_policy_logps_recomputed"].append(
            float(old_policy_recomputed)
        )

        actual_lengths = completion_attention_mask.sum(dim=1).float()
        unique_rewards: dict[Any, float] = {}
        grouped_rewards: dict[Any, list[float]] = defaultdict(list)
        for question_id, episode_id, reward in zip(question_ids, episode_ids, rewards.tolist(), strict=True):
            if episode_id not in unique_rewards:
                unique_rewards[episode_id] = reward
                grouped_rewards[question_id].append(reward)
        zero_std = [
            float(len(values) < 2 or np.std(values, ddof=1) == 0.0)
            for values in grouped_rewards.values()
        ]
        self.state.num_input_tokens_seen += int((prompt_mask.sum() + completion_attention_mask.sum()).item())
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]
        self._metrics[mode]["completions/mean_length"].append(actual_lengths.mean().item())
        self._metrics[mode]["completions/min_length"].append(actual_lengths.min().item())
        self._metrics[mode]["completions/max_length"].append(actual_lengths.max().item())
        self._metrics[mode]["completions/clipped_ratio"].append(
            sum(bool(output.extra_fields.get("overlong", False)) for output in outputs) / len(outputs)
        )
        episode_reward_mean = float(np.mean(list(unique_rewards.values())))
        self._metrics[mode]["reward"].append(episode_reward_mean)
        self._metrics[mode]["reward/score"].append(episode_reward_mean)
        self._metrics[mode]["reward_std"].append(float(np.mean([np.std(v, ddof=1) if len(v) > 1 else 0.0 for v in grouped_rewards.values()])))
        self._metrics[mode]["frac_reward_zero_std"].append(float(np.mean(zero_std)))
        self._metrics[mode]["agent/trajectories"].append(float(len(outputs)))
        self._metrics[mode]["agent/vllm_generate_calls"].append(float(batcher.generate_calls))
        self._metrics[mode]["agent/generated_tokens"].append(float(batcher.generated_tokens))
        self._metrics[mode]["agent/rollout_seconds"].append(float(rollout_seconds))
        self._metrics[mode]["agent/masked_rollouts"].append(float((optimization_mask == 0).sum().item()))

        # Branch streams repeat their parent episode's environment statistics.
        # Deduplicate by gen_uid so a branching policy cannot change the
        # reported task accuracy merely by emitting more trainable streams.
        unique_env_stats: dict[Any, dict[str, Any]] = {}
        for episode_id, output in zip(episode_ids, outputs, strict=True):
            unique_env_stats.setdefault(
                episode_id,
                output.extra_fields.get("env_stats", {}) or {},
            )
        env_stat_keys = {
            "math_correctness": "reward/correctness",
            "math_correctness_reward": "reward/correctness_reward",
            "math_format_valid": "reward/soft_format_valid",
            "math_format_reward": "reward/soft_format_reward",
            "consol_attempts": "graphrpo/controller_attempts",
            "consol_ops": "graphrpo/controller_valid_edits",
            "consol_controller_errors": "graphrpo/controller_errors",
            "graph_rpo_valid_edits": "graphrpo/valid_edits",
            "graph_rpo_creditable_edits": "graphrpo/creditable_edits",
            "graph_rpo_credited_edits": "graphrpo/credited_edits",
            "graph_rpo_scored_states": "graphrpo/scored_states",
            "graph_rpo_delta_abs_sum": "graphrpo/delta_abs_sum",
            "graph_rpo_counterfactual_probe_rollouts": (
                "graphrpo/counterfactual_probe_rollouts"
            ),
            "graph_rpo_counterfactual_tag_rate": (
                "graphrpo/counterfactual_tag_rate"
            ),
        }
        for source, metric in env_stat_keys.items():
            values = [
                float(stats.get(source, 0.0))
                for stats in unique_env_stats.values()
            ]
            self._metrics[mode][metric].append(float(np.mean(values)))

        prompt_text = self.processing_class.batch_decode(prompt_ids, skip_special_tokens=True)
        completion_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        self._logs["prompt"].extend(prompt_text)
        self._logs["completion"].extend(completion_text)
        self._logs["rewards"]["agent_reward"].extend(rewards.tolist())
        self._logs["advantages"].extend(
            ((advantages * policy_mask).sum(-1) / policy_mask.sum(-1).clamp(min=1)).tolist()
        )

        result = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_attention_mask": completion_attention_mask,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "old_per_token_logps": old_per_token_logps,
            "rollout_per_token_logps": rollout_logps,
        }
        if graph_loss_weights is not None:
            result["graphrpo_loss_weights"] = graph_loss_weights
        if fold_loss_weights is not None:
            result["foldgrpo_loss_weights"] = fold_loss_weights
        return result

    def _prepare_inputs(self, generation_batch):
        """Preserve every variable-count branch stream when splitting updates.

        QeRL's stock helper uses floor-sized slices because ordinary GRPO has
        a fixed output cardinality.  Agent episodes emit main plus zero or more
        branches, so floor slicing can silently discard remainder trajectories.
        ``tensor_split`` keeps all rows and permits adjacent microbatches to
        differ by one trajectory.
        """
        mode = "train" if self.model.training else "eval"
        if mode != "train":
            return self._generate_and_score_completions(generation_batch)

        generate_every = self.args.steps_per_generation * self.num_iterations
        if self._step % generate_every == 0 or self._buffered_inputs is None:
            started = time.perf_counter()
            generated = self._generate_and_score_completions(generation_batch)
            self._metrics[mode]["time/rollout"].append(time.perf_counter() - started)
            batch_size = len(next(value for value in generated.values() if value is not None))
            if batch_size < self.args.steps_per_generation:
                raise RuntimeError(
                    "agent trajectory count must be at least steps_per_generation; "
                    f"got {batch_size} < {self.args.steps_per_generation}"
                )
            permutation = torch.randperm(batch_size, device=self.accelerator.device)
            shuffled = {
                key: value[permutation] if value is not None else None
                for key, value in generated.items()
            }
            chunks = {
                key: torch.tensor_split(value, self.args.steps_per_generation, dim=0)
                if value is not None
                else [None] * self.args.steps_per_generation
                for key, value in shuffled.items()
            }
            self._buffered_inputs = [
                {key: values[index] for key, values in chunks.items()}
                for index in range(self.args.steps_per_generation)
            ]
        inputs = self._buffered_inputs[self._step % self.args.steps_per_generation]
        self._step += 1
        return inputs

    def _compute_loss(self, model: Any, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        prompt_ids = inputs["prompt_ids"]
        prompt_mask = inputs["prompt_mask"]
        completion_ids = inputs["completion_ids"]
        policy_mask = inputs["completion_mask"]
        completion_attention_mask = inputs["completion_attention_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_attention_mask], dim=1)
        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            completion_ids.size(1),
            compute_entropy=True,
        )
        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = (
            per_token_logps.detach()
            if old_per_token_logps is None
            else old_per_token_logps
        )
        log_ratio = per_token_logps - old_per_token_logps
        if self.importance_sampling_level == "token":
            log_importance_weights = log_ratio
        elif self.importance_sampling_level == "sequence":
            log_importance_weights = (
                (log_ratio * policy_mask).sum(-1) / policy_mask.sum(-1).clamp(min=1.0)
            ).unsqueeze(-1)
        else:
            raise ValueError(f"unknown importance_sampling_level: {self.importance_sampling_level}")

        ratios = torch.exp(log_importance_weights)
        clipped_ratios = torch.clamp(ratios, 1 - self.epsilon_low, 1 + self.epsilon_high)
        if self.args.delta is not None:
            ratios = torch.clamp(ratios, max=self.args.delta)
        advantages = inputs["advantages"]
        if advantages.ndim == 1:
            advantages = advantages.unsqueeze(1)
        per_token_loss = -torch.minimum(ratios * advantages, clipped_ratios * advantages)
        if self.top_entropy_quantile < 1.0:
            entropy_mask = self.get_high_entropy_mask(
                entropies, policy_mask, 1 - self.top_entropy_quantile
            )
            per_token_loss = per_token_loss * entropy_mask

        if "graphrpo_loss_weights" in inputs:
            loss = (
                per_token_loss * inputs["graphrpo_loss_weights"]
            ).sum() * float(self.args.steps_per_generation)
        elif "foldgrpo_loss_weights" in inputs:
            sequence_loss = (
                (per_token_loss * policy_mask).sum(-1)
                / policy_mask.sum(-1).clamp(min=1.0)
            )
            loss = (
                sequence_loss * inputs["foldgrpo_loss_weights"]
            ).sum() * float(self.args.steps_per_generation)
        elif self.loss_type == "grpo":
            loss = (
                (per_token_loss * policy_mask).sum(-1)
                / policy_mask.sum(-1).clamp(min=1.0)
            ).mean()
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * policy_mask).sum() / policy_mask.sum().clamp(min=1.0)
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * policy_mask).sum() / (
                per_token_loss.size(0) * self.max_completion_length
            )
        else:
            raise ValueError(f"unknown loss type: {self.loss_type}")
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite TRL agent policy loss")

        mode = "train" if model.training else "eval"
        token_count = policy_mask.sum().clamp(min=1.0)
        rollout_per_token_logps = inputs["rollout_per_token_logps"]
        rollout_prob_diff = (
            per_token_logps.detach() - rollout_per_token_logps
        ).abs()
        self._metrics[mode]["training/rollout_probs_diff_mean"].append(
            ((rollout_prob_diff * policy_mask).sum() / token_count).item()
        )
        self._metrics[mode]["training/rollout_probs_diff_max"].append(
            rollout_prob_diff.masked_fill(policy_mask == 0, 0.0).max().item()
        )
        self._metrics[mode]["actor/pg_loss"].append(loss.detach().item())
        mean_entropy = (entropies * policy_mask).sum() / token_count
        self._metrics[mode]["entropy"].append(
            self.accelerator.gather(mean_entropy).nanmean().item()
        )
        low = (ratios < 1 - self.epsilon_low) & (advantages < 0)
        high = (ratios > 1 + self.epsilon_high) & (advantages > 0)
        self._metrics[mode]["clip_ratio/low_mean"].append(
            ((low * policy_mask).sum() / token_count).item()
        )
        self._metrics[mode]["clip_ratio/high_mean"].append(
            ((high * policy_mask).sum() / token_count).item()
        )
        self._metrics[mode]["clip_ratio/region_mean"].append(
            (((low | high) * policy_mask).sum() / token_count).item()
        )
        return loss


__all__ = ["AgentGRPOTrainer"]
