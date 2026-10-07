# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import math
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pprint import pprint
from typing import Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.reference_graph_credit import (
    apply_reference_edit_credits,
    build_reference_scoring_batch,
    collect_reference_edit_plans,
    mean_answer_log_likelihood,
)
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.json_serialization import json_default
from verl.utils.metric import reduce_metrics
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger


_BATCH_LEVEL_REWARD_METRICS = {
    "avg_score",
    "std_score",
    "min_score",
    "max_score",
    "num_unique_gen_uids",
    "avg_trajs_per_gen_uid",
    "overlong_rate",
    "avg_num_turns",
}


def _scalarize_reward_extra_info(name, values):
    """Convert optional per-sample reward metadata to a scalar log value."""
    arr = np.asarray(values, dtype=object).reshape(-1)
    valid_values = [value for value in arr if value is not None]
    if not valid_values:
        return None

    try:
        if name in _BATCH_LEVEL_REWARD_METRICS:
            return float(valid_values[0])
        return float(np.asarray(valid_values, dtype=float).mean())
    except (TypeError, ValueError):
        return None


def _global_actor_mini_batch_size(config) -> int:
    """Return the rollout-expanded actor minibatch size on the controller.

    ``ppo_mini_batch_size`` is configured in prompt units. FSDP workers expand
    it by ``rollout.n`` before dividing it across data-parallel ranks, so the
    controller must use the same expanded size when deciding whether padding
    is required. The trainer node count is not part of this calculation.
    """
    prompt_mini_batch_size = int(config.actor_rollout_ref.actor.ppo_mini_batch_size)
    rollout_n = int(config.actor_rollout_ref.rollout.n)
    global_mini_batch_size = prompt_mini_batch_size * rollout_n
    if global_mini_batch_size <= 0:
        raise ValueError(
            "actor ppo_mini_batch_size and rollout.n must define a positive "
            f"global minibatch size, got {prompt_mini_batch_size} * {rollout_n}"
        )
    return global_mini_batch_size


def _actor_padding_divisor(config, world_size: int, loss_mode: str) -> int:
    """Return a padding divisor safe for both optimization and DP dispatch.

    Agent loops can emit a variable number of trajectories after branching.
    Vanilla PPO batches therefore need to be divisible by both the expanded
    actor minibatch and the actual data-parallel worker count before sequence
    balancing. GraphRPO handles minibatching separately but still requires an
    even DP split.
    """
    if world_size <= 0:
        raise ValueError(f"actor world_size must be positive, got {world_size}")
    if loss_mode == "graphrpo":
        return world_size
    return math.lcm(_global_actor_mini_batch_size(config), world_size)


def _without_dummy_trajectories(batch):
    """Return the real trajectories used for user-facing batch metrics."""
    dummy_gen_uid = batch.meta_info.get("gen_uid_dummy")
    gen_uids = batch.non_tensor_batch.get("gen_uid")
    if dummy_gen_uid is None or gen_uids is None:
        return batch
    keep_indices = [
        index
        for index, gen_uid in enumerate(gen_uids)
        if gen_uid != dummy_gen_uid and str(gen_uid) != str(dummy_gen_uid)
    ]
    return batch[keep_indices]


def _configured_rollout_workflow(config) -> str | None:
    """Return the optional project workflow without requiring plugin config.

    Upstream single-turn PPO/GRPO configurations do not define
    ``actor_rollout_ref.rollout.plugin``. ContextGraph launchers add it for
    their custom agent loops, so the shared trainer must treat it as optional.
    """
    rollout_config = config.actor_rollout_ref.rollout
    plugin = (
        rollout_config.get("plugin", None)
        if hasattr(rollout_config, "get")
        else getattr(rollout_config, "plugin", None)
    )
    if plugin is None:
        return None
    workflow = plugin.get("workflow", None) if hasattr(plugin, "get") else getattr(plugin, "workflow", None)
    return None if workflow is None else str(workflow)


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create Ray resource pools for distributed training.

        Initializes resource pools based on the resource pool specification,
        with each pool managing GPU resources across multiple nodes.
        For FSDP backend, uses max_colocate_count=1 to merge WorkerGroups.
        For Megatron backend, uses max_colocate_count>1 for different models.
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, using max_colocate_count=3: actor_critic_ref, rollout, reward model (optional)
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=3, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray._private.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.FOLDGRPO:
        # Initialize the mask for FOLDGRPO calculation
        token_level_rewards = data.batch["token_level_rewards"]
        foldgrpo_calculation_mask = data.batch["response_mask"]
        fix_bad_positive_adv = config.get("fix_bad_positive_adv", False)

        # Call compute_foldgrpo_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_foldgrpo_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=foldgrpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            gen_uid=data.non_tensor_batch["gen_uid"],  # Per-sample unique ID for de-duplicating retried rollouts
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            fix_bad_positive_adv=fix_bad_positive_adv,
            process_reward_mask=data.batch["process_reward_mask"],
            raw_token_level_scores=token_level_rewards if fix_bad_positive_adv else None,
            config=config,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.GRAPHRPO:
        excluded_gen_uids = set()
        if data.meta_info.get("gen_uid_dummy") is not None:
            excluded_gen_uids.add(data.meta_info["gen_uid_dummy"])
        advantages, returns = core_algos.compute_graphrpo_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            gen_uid=data.non_tensor_batch["gen_uid"],
            epsilon=float(config.get("graphrpo_epsilon", 1e-6)),
            process_reward_mask=data.batch.get("process_reward_mask"),
            graph_edit_credit_mask=data.batch.get("graph_edit_credit_mask"),
            graph_decision_mask=data.batch.get("graph_decision_mask"),
            excluded_gen_uids=excluded_gen_uids,
            config=config,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        data.batch["graphrpo_loss_weights"] = core_algos.compute_graphrpo_loss_weights(
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            gen_uid=data.non_tensor_batch["gen_uid"],
            excluded_gen_uids=excluded_gen_uids,
            allow_empty=bool(config.get("graphrpo_memory_only", False)),
        )
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data

# TODO@Miao: Add plugin configs
class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping or Role.ActorRolloutRef in role_worker_mapping, (
                f"{role_worker_mapping.keys()=}"
            )

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.role_worker_mapping)
        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = (
            config.actor_rollout_ref.model.get("lora_rank", 0) > 0
            or config.actor_rollout_ref.model.get("lora_adapter_path") is not None
        )

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("train_max_samples", -1),
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("val_max_samples", -1),
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(
        self,
        inputs,
        outputs,
        gts,
        scores,
        reward_extra_infos_dict,
        dump_path,
        trajectory_fields=None,
    ):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        # Agent-loop outputs contain the role-preserving conversation and
        # executor statistics needed to turn successful validation rollouts
        # into multi-turn SFT data.  Keep these optional so ordinary rollout
        # implementations and existing dump consumers remain compatible.
        for k, v in (trajectory_fields or {}).items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False, default=json_default))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _skip_empty_memory_batch(self, batch, metrics, timing_raw):
        """No trainable M tokens: log the audit without invoking model workers."""
        if not self.config.algorithm.get("graphrpo_memory_only", False):
            return False
        if batch.batch["response_mask"].any():
            return False
        metrics["training/skipped_empty_memory_batch"] = 1
        metrics["training/actor_update_calls"] = 0
        stats = batch.non_tensor_batch.get("env_stats", [])
        metrics["training/continuation_failed_groups"] = sum(
            (row or {}).get("graph_rpo_continuation_failed", 0) for row in stats
        )
        batch.batch["token_level_scores"] = torch.zeros_like(batch.batch["response_mask"], dtype=torch.float32)
        rollout_data_dir = self.config.trainer.get("rollout_data_dir")
        if rollout_data_dir:
            self._log_rollout_data(batch, {}, timing_raw, rollout_data_dir)
        return True

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_dict.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )

            trajectory_fields: dict[str, list] = {}
            for key in (
                "graph_rpo_continuation",
                "messages", "env_stats", "is_finish", "termination_reason",
                "agent_name", "graph_trace", "graph_state", "graph_rewards",
                "model_contexts", "branch_model_contexts", "contextgraph_memory_mode",
                "judge_audit", "uid", "gen_uid", "diagnostic_fix", "repeat_advice_count",
            ):
                values = batch.non_tensor_batch.get(key)
                if values is not None and len(values) == len(scores):
                    trajectory_fields[key] = (
                        values.tolist() if hasattr(values, "tolist") else list(values)
                    )

            extra_infos = batch.non_tensor_batch.get("extra_info")
            if extra_infos is not None and len(extra_infos) == len(scores):
                normalized_extra_infos = (
                    extra_infos.tolist()
                    if hasattr(extra_infos, "tolist")
                    else list(extra_infos)
                )
                trajectory_fields["task_id"] = [
                    str(info.get("task_id", info.get("instance_id", "unknown")))
                    if isinstance(info, dict) else "unknown"
                    for info in normalized_extra_infos
                ]
                trajectory_fields["question"] = [
                    str(info.get("query", info.get("problem_statement", "")))
                    if isinstance(info, dict) else ""
                    for info in normalized_extra_infos
                ]

            # reward_extra_infos_dict contains episode-deduplicated values for
            # logging. Preserve the binary label of each concrete rollout for
            # judge auditing and graph-evaluator dataset construction.
            env_stats = trajectory_fields.get("env_stats")
            if env_stats is not None:
                trajectory_fields["task_reward"] = [
                    stats.get("task_reward") if isinstance(stats, dict) else None
                    for stats in env_stats
                ]

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
                trajectory_fields=trajectory_fields,
            )

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []
        trajectory_fields: dict[str, list] = defaultdict(list)

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # Validation sampling must use the frozen validation protocol rather
            # than inheriting the training-rollout temperature.  This matters
            # when collecting multiple stochastic target-policy trajectories
            # for a graph evaluator.
            test_batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.val_kwargs.temperature

            # Add uid to batch (aligned with fit() - always create new uids)
            test_batch.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
            )

            test_batch.meta_info["global_steps"] = self.global_steps

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # Add gen_uid to batch after repeat (aligned with fit())
            test_batch.non_tensor_batch["gen_uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
            )

            # Project agent loops require a workflow; upstream single-turn
            # PPO/GRPO intentionally has no rollout.plugin section.
            workflow = _configured_rollout_workflow(self.config)
            if workflow is not None:
                test_batch.non_tensor_batch["workflow"] = np.full(
                    len(test_batch.batch), workflow, dtype=object
                )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Set meta_info for generation (aligned with fit())
            test_batch.meta_info.update({
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            })
            print(f"test_batch meta info: {test_batch.meta_info}")

            # Generate sequences (aligned with fit() - no padding before generation)
            if not self.async_rollout_mode:
                test_batch = self.actor_rollout_wg.generate_sequences(test_batch)
            else:
                test_batch = self.async_rollout_manager.generate_sequences(test_batch)

            # Update timing from meta_info if available (aligned with fit())
            if "timing" in test_batch.meta_info:
                test_batch.meta_info.pop("timing", None)
            test_batch.meta_info["global_steps"] = self.global_steps

            print("validation generation end")

            # Agent loops may expand one episode into several training segments.
            # Align every logged/metric row with the generated batch, not prompts.
            sample_uids.extend(test_batch.non_tensor_batch["uid"])
            sample_inputs.extend(self.tokenizer.decode(ids, skip_special_tokens=True)
                                 for ids in test_batch.batch["input_ids"])
            sample_gts.extend(item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None)
                              for item in test_batch)

            # Compute response_mask if not present (aligned with fit())
            if "response_mask" not in test_batch.batch.keys():
                test_batch.batch["response_mask"] = compute_response_mask(test_batch)

            # Store generated outputs
            output_ids = test_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch.meta_info["validate"] = True

            # Evaluate using reward_function (aligned with fit() using compute_reward)
            if self.val_reward_fn is None:
                raise ValueError("val_reward_fn must be provided for validation.")
            reward_tensor, reward_extra_info = compute_reward(test_batch, self.val_reward_fn)
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            if reward_extra_info:
                for key, lst in reward_extra_info.items():
                    reward_extra_infos_dict[key].extend(lst)

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            for key in (
                "messages", "env_stats", "is_finish", "termination_reason",
                "agent_name", "graph_trace", "graph_state", "graph_rewards",
                "model_contexts", "branch_model_contexts", "contextgraph_memory_mode",
                "judge_audit", "uid", "gen_uid", "diagnostic_fix", "repeat_advice_count",
            ):
                values = test_batch.non_tensor_batch.get(key)
                if values is not None and len(values) == len(scores):
                    trajectory_fields[key].extend(values.tolist() if hasattr(values, "tolist") else list(values))

            extra_infos = test_batch.non_tensor_batch.get("extra_info")
            if extra_infos is not None and len(extra_infos) == len(scores):
                normalized_extra_infos = (
                    extra_infos.tolist()
                    if hasattr(extra_infos, "tolist")
                    else list(extra_infos)
                )
                trajectory_fields["task_id"].extend(
                    str(info.get("task_id", info.get("instance_id", "unknown")))
                    if isinstance(info, dict) else "unknown"
                    for info in normalized_extra_infos
                )
                trajectory_fields["question"].extend(
                    str(info.get("query", info.get("problem_statement", "")))
                    if isinstance(info, dict) else ""
                    for info in normalized_extra_infos
                )

            # Validation reward metrics are episode-deduplicated and therefore
            # may contain a batch mean repeated for every row. Persist the
            # concrete terminal outcome from each trajectory instead.
            env_stats = test_batch.non_tensor_batch.get("env_stats")
            if env_stats is not None and len(env_stats) == len(scores):
                normalized_env_stats = (
                    env_stats.tolist() if hasattr(env_stats, "tolist") else list(env_stats)
                )
                trajectory_fields["task_reward"].extend(
                    stats.get("task_reward") if isinstance(stats, dict) else None
                    for stats in normalized_env_stats
                )

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
                trajectory_fields=trajectory_fields,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        # Add scalarized reward_extra_info metrics with val/ prefix (aligned with fit())
        if reward_extra_infos_dict:
            for key, val in reward_extra_infos_dict.items():
                scalar_val = _scalarize_reward_extra_info(key, val)
                if scalar_val is not None:
                    metric_dict[f"val/{key}"] = scalar_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[actor_role],
                config=self.config.actor_rollout_ref,
                role=str(actor_role),
            )
            self.resource_pool_to_cls[resource_pool][str(actor_role)] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cfg = omega_conf_to_dataclass(self.config.critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy and Role.RefPolicy in self.role_worker_mapping:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool][str(Role.RewardModel)] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            if str(Role.RefPolicy) in all_wg:
                self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
                self.ref_policy_wg.init_model()
            else:
                # Model engine: ActorRolloutRefWorker
                assert str(Role.ActorRolloutRef) in all_wg, f"{all_wg.keys()=}"
                self.ref_policy_wg = all_wg[str(Role.ActorRolloutRef)]

        self.rm_wg = None
        # initalization of rm_wg will be deprecated in the future
        if self.use_rm:
            self.rm_wg = all_wg[str(Role.RewardModel)]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg[str(actor_role)]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager

            self.async_rollout_mode = True
            if self.config.reward_model.enable and self.config.reward_model.enable_resource_pool:
                rm_resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            else:
                rm_resource_pool = None

            self.async_rollout_manager = AgentLoopManager(
                config=self.config,
                worker_group=self.actor_rollout_wg,
                rm_resource_pool=rm_resource_pool,
            )

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        if (
            hasattr(self.config.actor_rollout_ref.actor.checkpoint, "async_save")
            and self.config.actor_rollout_ref.actor.checkpoint.async_save
        ) or (
            "async_save" in self.config.actor_rollout_ref.actor.checkpoint
            and self.config.actor_rollout_ref.actor.checkpoint["async_save"]
        ):
            print("skip write latest_checkpointed_iteration.txt when async_save is True")
            return
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)
            if self.use_rm:
                self.rm_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm:
                self.rm_wg.stop_profile()

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1)  # (train_batch_size,)
        workload_lst = calculate_workload(global_seqlen_lst)
        world_size = self.actor_rollout_wg.world_size
        if keep_minibatch:
            # Decouple the DP balancing and mini-batching.
            minibatch_size = self.config.actor_rollout_ref.actor.get("ppo_mini_batch_size")
            minibatch_num = len(workload_lst) // minibatch_size
            global_partition_lst = [[] for _ in range(world_size)]
            for i in range(minibatch_num):
                rearrange_minibatch_lst = get_seqlen_balanced_partitions(
                    workload_lst[i * minibatch_size : (i + 1) * minibatch_size],
                    k_partitions=world_size,
                    equal_size=True,
                )
                for j, part in enumerate(rearrange_minibatch_lst):
                    global_partition_lst[j].extend([x + minibatch_size * i for x in part])
        else:
            global_partition_lst = get_seqlen_balanced_partitions(
                workload_lst, k_partitions=world_size, equal_size=True
            )
        # Place smaller micro-batches at both ends to reduce the bubbles in pipeline parallel.
        for idx, partition in enumerate(global_partition_lst):
            partition.sort(key=lambda x: (workload_lst[x], x))
            ordered_partition = partition[::2] + partition[1::2][::-1]
            global_partition_lst[idx] = ordered_partition
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _pad_dummy_sample(self, batch: DataProto, divisor: int | None = None):
        """Pad dummy sample to make batch size divisible by mini_batch_size"""
        divisor = divisor or _global_actor_mini_batch_size(self.config)
        num_padding = (-len(batch)) % divisor
        if num_padding == 0:
            return batch, None
        print(f"pad batch size from {len(batch)} to {len(batch) + num_padding}")
        assert 'response_mask' in batch.batch, "response_mask is required for defining padding dummy sample"
        dummy_sample = deepcopy(batch[0:1])
        # Dummy tokens must not contribute to policy, entropy, or KL losses.
        # Keep this zero independently of overlong_mask as defense in depth.
        dummy_sample.batch["response_mask"] = torch.zeros_like(dummy_sample.batch["response_mask"])
        if "overlong_mask" in dummy_sample.batch:
            dummy_sample.batch["overlong_mask"] = torch.zeros_like(dummy_sample.batch["overlong_mask"])
        dummy_sample.batch["attention_mask"] = torch.zeros_like(dummy_sample.batch["attention_mask"])
        dummy_sample.non_tensor_batch["uid"] = np.array([uuid.uuid4(),], dtype=object)
        gen_uid_dummy = uuid.uuid4()
        dummy_sample.non_tensor_batch["gen_uid"] = np.array([gen_uid_dummy,], dtype=object)
        batch = DataProto.concat([batch] + [dummy_sample for _ in range(num_padding)])
        batch.meta_info["gen_uid_dummy"] = gen_uid_dummy
        return batch, gen_uid_dummy

    def _answer_likelihood_graph_credit_backend(self) -> str | None:
        if str(self.config.algorithm.adv_estimator).lower() != "graphrpo":
            return None
        plugin = self.config.actor_rollout_ref.rollout.plugin
        backend = str(plugin.get("graph_rpo_credit_backend", "external_evaluator")).lower()
        if backend in {
            "reference_answer_likelihood",
            "old_policy_answer_likelihood",
        }:
            return backend
        return None

    def _compute_answer_likelihood_graph_credit(
        self,
        batch: DataProto,
        backend: str,
    ) -> dict[str, float | int]:
        """Score correct answers before actor update and attach edit-local deltas."""
        if backend == "reference_answer_likelihood" and not self.use_reference_policy:
            raise RuntimeError(
                "reference-answer GraphRPO requires an initialized frozen reference policy"
            )
        if backend not in {
            "reference_answer_likelihood",
            "old_policy_answer_likelihood",
        }:
            raise ValueError(f"unsupported answer-likelihood GraphRPO backend: {backend}")
        metric_namespace = (
            "reference" if backend == "reference_answer_likelihood" else "old_policy"
        )
        metric_prefix = f"graphrpo/{metric_namespace}"
        plans, keys = collect_reference_edit_plans(batch)
        if not keys:
            batch.batch["graph_edit_credit_mask"] = torch.zeros_like(
                batch.batch["response_mask"], dtype=torch.float32
            )
            return {
                f"{metric_prefix}_creditable_edits": 0,
                f"{metric_prefix}_scored_states": 0,
                f"{metric_prefix}_delta_sum": 0.0,
                f"{metric_prefix}_delta_abs_sum": 0.0,
                f"{metric_prefix}_delta_mean": 0.0,
            }

        plugin = self.config.actor_rollout_ref.rollout.plugin
        scoring_batch, answer_mask = build_reference_scoring_batch(
            self.tokenizer,
            keys,
            max_prompt_length=int(
                plugin.get(
                    "graph_rpo_reference_max_prompt_length",
                    self.config.actor_rollout_ref.rollout.prompt_length,
                )
            ),
            max_answer_length=int(plugin.get("graph_rpo_reference_max_answer_length", 128)),
            enable_thinking=bool(plugin.get("graph_rpo_reference_enable_thinking", False)),
        )
        if backend == "reference_answer_likelihood":
            scoring_worker = self.actor_rollout_wg if self.ref_in_actor else self.ref_policy_wg
        else:
            # This call happens after rollout old-log-prob recomputation and before
            # update_actor. compute_answer_log_prob runs under no_grad in the actor
            # worker, so these batch-local utility values are detached from optimization.
            scoring_worker = self.actor_rollout_wg
        scoring_batch_padded, pad_size = pad_dataproto_to_divisor(
            scoring_batch,
            scoring_worker.world_size,
        )
        if backend == "reference_answer_likelihood":
            score_output = scoring_worker.compute_ref_log_prob(scoring_batch_padded)
            score_key = "ref_log_prob"
        else:
            score_output = scoring_worker.compute_answer_log_prob(scoring_batch_padded)
            score_key = "answer_log_probs"
        score_output = unpad_dataproto(score_output, pad_size)
        likelihood_values = mean_answer_log_likelihood(
            score_output.batch[score_key],
            answer_mask,
        )
        likelihoods = dict(zip(keys, likelihood_values, strict=True))
        raw_costs = plugin.get("graph_rpo_operation_costs", {}) or {}
        return apply_reference_edit_credits(
            batch,
            plans,
            likelihoods,
            delta_max=float(plugin.get("graph_rpo_delta_max", 1.0)),
            delta_scale=float(plugin.get("graph_rpo_delta_scale", 1.0)),
            operation_costs=dict(raw_costs),
            credit_backend=backend,
            metric_namespace=metric_namespace,
        )

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        current_epoch = self.global_steps // len(self.train_dataloader)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                logger.finish()
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

                # add uid to batch
                import numpy as np
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                batch.meta_info["global_steps"] = self.global_steps
                batch = batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                # TODO@Miao[DONE]: add uid for each sample in batch. The key to the uid need to be different from "uid" 
                # - "uid" -> question level (unique id for each question, repeated by self.config.actor_rollout_ref.rollout.n)
                # - "gen_uid" -> sample level (unique id for each sample)

                batch.non_tensor_batch["gen_uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                workflow = _configured_rollout_workflow(self.config)
                if workflow is not None:
                    batch.non_tensor_batch["workflow"] = np.full(
                        len(batch.batch), workflow, dtype=object
                    )

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            batch = self.actor_rollout_wg.generate_sequences(batch)
                        else:
                            # TODO@Miao: revise fold_agent_loop.py
                            batch = self.async_rollout_manager.generate_sequences(batch)

                        timing_raw.update(batch.meta_info["timing"])
                        batch.meta_info.pop("timing", None)
                        batch.meta_info["global_steps"] = self.global_steps

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)

                    if self._skip_empty_memory_batch(batch, metrics, timing_raw):
                        # Count the rollout attempt, not an optimizer update. Do not
                        # run AdamW/scheduler or empty-tensor training metrics.
                        metrics.update({"training/global_step": self.global_steps,
                                        "training/epoch": epoch})
                        logger.log(data=metrics, step=self.global_steps)
                        self._stop_profiling(curr_step_profile)
                        prev_step_profile = False
                        progress_bar.update(1)
                        self.global_steps += 1
                        curr_step_profile = self.global_steps in (self.config.global_profiler.steps or [])
                        if is_last_step:
                            progress_bar.close()
                            logger.finish()
                            return
                        if hasattr(self.train_dataset, "on_batch_end"):
                            self.train_dataset.on_batch_end(batch=batch)
                        continue

                    # mask_rollout is an optional optimization decision supplied
                    # by ContextGraph agent loops. Standard single-turn rollouts
                    # do not emit it and should keep every generated response.
                    if self.config.algorithm.mask_overlong:
                        optimization_masked_rollouts = 0
                        if "mask_rollout" in batch.batch:
                            mask_rollout = batch.batch["mask_rollout"]
                            batch.batch["overlong_mask"] = (~mask_rollout.bool()).int()
                            optimization_masked_rollouts = mask_rollout.sum().item()
                        metrics['optimization_masked_rollouts'] = optimization_masked_rollouts

                    # TODO@Miao[DONE]: pad batch size to muliplicative of mini_batch_size
                    loss_mode = self.config.actor_rollout_ref.actor.policy_loss.get(
                        "loss_mode", "vanilla"
                    )
                    padding_divisor = _actor_padding_divisor(
                        self.config,
                        world_size=self.actor_rollout_wg.world_size,
                        loss_mode=loss_mode,
                    )
                    if len(batch) % padding_divisor != 0:
                        batch, gen_uid_dummy = self._pad_dummy_sample(
                            batch, divisor=padding_divisor
                        )

                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(
                                data=batch, config=self.config, tokenizer=self.tokenizer
                            )
                        else:
                            # TODO@Miao: implement the reward_fn
                            # - pay attention to the multiple trajectories of a rollout when calculating metrics
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
                            if os.environ.get("VERL_VERBOSE_DIAGNOSTICS") == "1":
                                print(reward_tensor)

                    # Operating Mode Selection:
                    # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
                    # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
                    #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                    if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                        from verl.trainer.ppo.rollout_corr_helper import apply_rollout_correction

                        apply_rollout_correction(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    else:  # Recompute old_log_probs
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            actor_config = self.config.actor_rollout_ref.actor
                            entropy_agg = agg_loss(
                                loss_mat=entropys,
                                loss_mask=response_masks,
                                loss_agg_mode=actor_config.loss_agg_mode,
                                loss_scale_factor=actor_config.loss_scale_factor,
                            )
                            old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            batch = batch.union(old_log_prob)
                            if "rollout_log_probs" in batch.batch.keys():
                                # TODO: we may want to add diff of probs too.
                                from verl.utils.debug.metrics import calculate_debug_metrics

                                metrics.update(calculate_debug_metrics(batch))

                    assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    answer_likelihood_backend = self._answer_likelihood_graph_credit_backend()
                    if answer_likelihood_backend is not None:
                        with marked_timer("answer_likelihood_graph_credit", timing_raw, color="olive"):
                            metrics.update(
                                self._compute_answer_likelihood_graph_credit(
                                    batch,
                                    answer_likelihood_backend,
                                )
                            )

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Compute rollout correction: IS weights, rejection sampling, and metrics
                        # Only runs in decoupled mode (computes once per batch using stable π_old)
                        # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
                        if (
                            rollout_corr_config is not None
                            and "rollout_log_probs" in batch.batch
                            and not bypass_recomputing_logprobs  # Only in decoupled mode
                        ):
                            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                            # Compute IS weights, apply rejection sampling, compute metrics
                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                            # IS and off-policy metrics already have rollout_corr/ prefix
                            metrics.update(is_metrics)

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        # TODO@Miao[Done]: revise multiple trajectory advantage calculation
                        # TODO@Miao[Done]: add algorithm.fix_bad_positive_adv config
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            rollout_config = self.config.actor_rollout_ref.rollout
                            batch.meta_info["multi_turn"] = rollout_config.multi_turn.enable
                            # TODO: Make "temperature" single source of truth from generation.
                            batch.meta_info["temperature"] = rollout_config.temperature

                            # TODO@Miao[Done]: revise policy gradient loss calculation to include overlong masking
                            # - Pay attention to the influence of dummy samples when doing normalization
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)
                        if self.config.algorithm.get("graphrpo_memory_only", False):
                            metrics["training/actor_update_calls"] = 1
                            metrics["training/skipped_empty_memory_batch"] = 0

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                # Check if the conditions for saving a checkpoint are met.
                # The conditions include a mandatory condition (1) and
                # one of the following optional conditions (2/3/4):
                # 1. The save frequency is set to a positive value.
                # 2. It's the last training step.
                # 3. The current step number is a multiple of the save frequency.
                # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )

                # TODO@Miao[Done]: update metrics from reward_extra_infos_dict (where we calculated metrics deduplicated by gen_uid)
                if reward_extra_infos_dict:
                    for key, val in reward_extra_infos_dict.items():
                        scalar_val = _scalarize_reward_extra_info(key, val)
                        if scalar_val is not None:
                            metrics[f"reward/{key}"] = scalar_val
                            
                # collect metrics
                metrics_batch = _without_dummy_trajectories(batch)
                metrics.update(compute_data_metrics(batch=metrics_batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    logger.finish()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)
