"""Exercise the production Ray-init path without loading GPU trainer modules."""
import ast
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest

from verl.trainer.constants_ppo import build_ppo_ray_init_kwargs


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("env_mode", ["absent", "empty", "gram", "override"])
@pytest.mark.parametrize("transfer_queue", [False, True])
def test_real_run_ppo_initializes_ray_from_strict_config_without_mutation(monkeypatch, env_mode, transfer_queue):
    monkeypatch.setenv("LOCAL_SEARCH_URL", "http://search:18999")
    monkeypatch.delenv("VERL_VERBOSE_DIAGNOSTICS", raising=False)
    overrides = [f"transfer_queue.enable={transfer_queue}"]
    if env_mode == "empty":
        overrides += ["+ray_kwargs.ray_init.runtime_env.env_vars={}"]
    elif env_mode in {"gram", "override"}:
        overrides += ["+ray_kwargs.ray_init.runtime_env.env_vars.GRAM_TRACE_DIR=/smoke/traces",
                      "+ray_kwargs.ray_init.runtime_env.env_vars.JUDGE_MODEL=gpt-5-nano"]
        if env_mode == "override":
            overrides += ["+ray_kwargs.ray_init.runtime_env.env_vars.VLLM_USE_V1=0",
                          "+ray_kwargs.ray_init.runtime_env.env_vars.NCCL_DEBUG=INFO"]
    with initialize_config_dir(config_dir=str(ROOT / "verl/trainer/config"), version_base=None):
        config = compose(config_name="ppo_trainer", overrides=overrides)
    # Stronger than Hydra struct mode: the original config is also read-only.
    OmegaConf.set_readonly(config, True)
    before = OmegaConf.to_container(config, resolve=False)
    fake_ray = Mock()
    fake_ray.is_initialized.return_value = False
    runner = Mock()
    task_runner = SimpleNamespace(remote=Mock(return_value=runner))
    source = ROOT / "verl/trainer/main_ppo.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_ppo")
    namespace = {"ray": fake_ray, "os": os, "OmegaConf": OmegaConf,
                 "build_ppo_ray_init_kwargs": build_ppo_ray_init_kwargs,
                 "is_cuda_available": False}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    namespace["run_ppo"](config, task_runner_class=task_runner)
    fake_ray.init.assert_called_once()
    env = fake_ray.init.call_args.kwargs["runtime_env"]["env_vars"]
    assert env["VLLM_USE_V1"] == "1"
    assert env.get("TRANSFER_QUEUE_ENABLE") == ("1" if transfer_queue else None)
    assert env["LOCAL_SEARCH_URL"] == "http://search:18999"
    if env_mode in {"gram", "override"}:
        assert env["GRAM_TRACE_DIR"] == "/smoke/traces"
        assert env["JUDGE_MODEL"] == "gpt-5-nano"
    if env_mode == "override":
        assert env["NCCL_DEBUG"] == "INFO"
    assert OmegaConf.to_container(config, resolve=False) == before
    runner.run.remote.assert_called_once_with(config)
    fake_ray.get.assert_called_once_with(runner.run.remote.return_value)
