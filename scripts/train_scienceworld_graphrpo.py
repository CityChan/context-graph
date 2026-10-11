"""Resolve, validate and record the ScienceWorld v3_long training protocol.

The prepare-only mode needs Hydra but does not import Ray, Torch, or a simulator.
Actual execution uses the existing registered ContextGraph/VERL training loop.
"""

import json
import os
from pathlib import Path
import subprocess
import sys

import hydra
from omegaconf import OmegaConf

from agents.scienceworld_training import validate_reward_profile


def validate_config(config):
    rollout = config.actor_rollout_ref.rollout
    plugin = rollout.plugin
    validate_reward_profile(plugin.scienceworld_training_reward_profile,
                            ability="ScienceWorld@real", algorithm=config.algorithm, is_train=True)
    if plugin.scienceworld_training_reward_profile != "v3_long":
        raise ValueError("This launcher requires the v3_long reward profile")
    if plugin.scienceworld_prompt_profile != "legacy":
        raise ValueError("v3_long uses legacy prompts in both training and validation")
    if plugin.graph_rpo_credit_backend != "outcome_only":
        raise ValueError("v3_long uses outcome_only credit")
    if plugin.final_answer_reserve or plugin.must_finish:
        raise ValueError("ScienceWorld uses environment termination; disable finalizer/must_finish")
    if plugin.graph_rpo_scope_process_reward or list(plugin.process_reward) != ["graphrpo"]:
        raise ValueError("v3_long excludes scope, compress and flat reward shaping")
    if config.algorithm.mask_overlong or config.algorithm.graphrpo_norm_adv_by_std:
        raise ValueError("v3_long retains exhausted episodes and uses mean-centered advantages")
    if (rollout.agent.default_agent_loop != "context_graph_isolated_agent"
            or plugin.get("graph_action_protocol") != "explicit_tools"
            or plugin.structured_graph_controller):
        raise ValueError("v3_long requires isolated GraphRPO with explicit_tools")
    if int(plugin.max_traj) < max(int(plugin.max_session), int(plugin.val_max_session)) + 1:
        raise ValueError("max_traj must retain the main stream and every possible branch")
    if int(rollout.n) < 2 or not 1 <= int(config.algorithm.graphrpo_dynamic_min_groups) <= int(config.data.train_batch_size):
        raise ValueError("Invalid candidate/group counts")
    if int(config.actor_rollout_ref.actor.ppo_epochs) != 1 or config.actor_rollout_ref.actor.use_rollout_log_probs:
        raise ValueError("v3_long requires one on-policy optimizer epoch")
    if float(plugin.graph_controller_temperature) != float(rollout.temperature) or float(plugin.graph_controller_top_p) != 1.0:
        raise ValueError("Controller sampling must match the on-policy temperature with top_p=1")
    if int(rollout.max_model_len) < int(rollout.prompt_length) + int(rollout.response_length):
        raise ValueError("max_model_len must cover the configured working context")


def write_manifest(config):
    validate_config(config)
    # Reject a missing backbone before creating a misleading ready manifest.
    if OmegaConf.is_missing(config.actor_rollout_ref.model, "path"):
        raise ValueError("Set actor_rollout_ref.model.path to the intended training checkpoint")
    root = Path(__file__).resolve().parents[1]
    kwargs = {"cwd": root, "text": True, "creationflags": subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0}
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], **kwargs).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], **kwargs).splitlines()
    destination = Path(config.trainer.default_local_dir)
    destination.mkdir(parents=True, exist_ok=True)
    # New executions get a unique record, including resumes/overrides.
    from datetime import datetime, timezone
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    prefix = destination / f"protocol-{stamp}"
    OmegaConf.save(config, str(prefix) + ".yaml", resolve=True)
    metadata = {
        "profile": "scienceworld_graphrpo_v3_long", "commit": revision,
        "graph_action_protocol": config.actor_rollout_ref.rollout.plugin.graph_action_protocol,
        "graph_protocol_revision": "model_owned_explicit_tools_v1",
        "dirty_paths": dirty, "python": sys.executable, "argv": sys.argv[1:],
        "prepare_only": bool(config.scienceworld_profile_prepare_only),
        "validation_reward": "native binary completion; raw and peak environment scores retained",
        "source": "screenshot-derived objective; source branch was not available for a code comparison",
        "config": str(prefix) + ".yaml",
    }
    Path(str(prefix) + ".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf8")
    print("SCIENCEWORLD_RL_PROFILE " + json.dumps(metadata), flush=True)


@hydra.main(config_path="../verl/trainer/config", config_name="scienceworld_graphrpo_v3_long", version_base=None)
def main(config):
    write_manifest(config)
    if config.scienceworld_profile_prepare_only:
        return
    from verl.utils.config import validate_config as validate_verl_config
    validate_verl_config(config, use_reference_policy=True, use_critic=False)
    # Import registers context_graph_isolated_agent before workers start.
    import scripts.train_graph  # noqa: F401
    from verl.trainer.main_ppo import run_ppo
    run_ppo(config)


if __name__ == "__main__":
    main()
