"""Regression tests for startup failures seen in jobs 1031462 and 1031463."""
import ast
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("first", ["scripts.train_fold", "scripts.train_graph", "scripts.train_baseline",
                                   "verl.experimental.agent_loop"])
def test_training_entrypoint_import_order_keeps_all_registrations(first, tmp_path):
    # Real package initializer + real training modules; stub only heavy runtime
    # dependencies. Every order runs in a fresh interpreter, as on Ray workers.
    script = tmp_path / "probe.py"
    script.write_text('''import importlib, sys, types
from pathlib import Path
root = Path(sys.argv[1])
sys.path.insert(0, str(root))
def module(name, **attributes):
    value = types.ModuleType(name)
    value.__dict__.update(attributes)
    sys.modules[name] = value
    return value
verl = module("verl", DataProto=object)
verl.__path__ = [str(root / "verl")]
registry = {}
def register(name):
    def decorate(cls):
        registry[name] = cls
        return cls
    return decorate
module("verl.experimental.agent_loop.agent_loop", AgentLoopBase=object, AgentLoopOutput=object,
       AgentLoopManager=object, AgentLoopWorker=object, AsyncLLMServerManager=object, register=register)
module("verl.experimental.agent_loop.single_turn_agent_loop", SingleTurnAgentLoop=object)
module("verl.experimental.agent_loop.tool_agent_loop", ToolAgentLoop=object)
module("verl.experimental.agent_loop.code_agent_loop", ReactAgentCodeLoop=object,
       FoldAgentCodeLoop=object, ContextGraphCodeIsolatedLoop=object)
module("agents.utils", CallLLM=object, TaskContext=object)
for name in ("fold_agent", "graph_agent", "graph_agent_isolated", "react_agent"):
    module("agents." + name, process_item=object)
importlib.import_module(sys.argv[2])
package = importlib.import_module("verl.experimental.agent_loop")
assert {"fold_agent", "context_graph_agent", "context_graph_isolated_agent", "react_agent"} <= registry.keys(), registry
assert package.FoldAgentLoop is registry["fold_agent"]
assert package.ContextGraphAgentLoop is registry["context_graph_agent"]
assert package.ReactAgentLoop is registry["react_agent"]
''', encoding="utf8")
    result = subprocess.run([sys.executable, str(script), str(ROOT), first], cwd=ROOT,
        capture_output=True, text=True, timeout=30,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    assert result.returncode == 0, result.stdout + result.stderr


def stub_module(monkeypatch, name, **attrs):
    module = ModuleType(name)
    module.__dict__.update(attrs)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def load_file(monkeypatch, name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def test_import_helpers_does_not_require_removed_lora_or_fp8_classes(monkeypatch):
    stub_module(monkeypatch, "vllm", __version__="0.27.1")
    stub_module(monkeypatch, "vllm.lora.request", LoRARequest=type("LoRARequest", (), {}))
    stub_module(monkeypatch, "msgspec", field=lambda **kwargs: None)
    stub_module(monkeypatch, "verl.third_party.vllm", get_version=lambda name: "0.27.1")
    stub_module(monkeypatch, "torch", bfloat16=object())
    module = load_file(monkeypatch, "lora_import_probe", "verl/utils/vllm/utils.py")
    assert module.is_version_ge(minver="0.12")
    fp8 = load_file(monkeypatch, "fp8_import_probe", "verl/utils/vllm/vllm_fp8_utils.py")
    # Selection of the BF16 path still works without importing the removed MoE
    # implementation. Attempting actual FP8 use must still fail explicitly.
    class FP8Config:
        pass
    stub_module(monkeypatch, "vllm.model_executor.layers.quantization.fp8", Fp8Config=FP8Config)
    assert not fp8.is_fp8_model(SimpleNamespace(quant_config=None))
    with pytest.raises(ImportError):
        fp8.is_fp8_weight("x.weight", object())


@pytest.mark.parametrize("lora_enabled", [False, True])
def test_worker_patches_lora_only_when_enabled(lora_enabled, monkeypatch):
    # Execute the actual method with fake process/device/worker objects; no GPU.
    tree = ast.parse((ROOT / "verl/workers/rollout/vllm_rollout/vllm_rollout.py").read_text(encoding="utf8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "vLLMAsyncRollout")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_init_worker")
    hijack, worker = Mock(), Mock()
    namespace = {"torch": SimpleNamespace(distributed=SimpleNamespace(is_initialized=lambda: True), bfloat16="bf16"),
        "Any": object, "os": os, "is_npu_available": False, "ray_noset_visible_devices": lambda: False,
        "VLLMHijack": SimpleNamespace(hijack=hijack), "is_version_ge": lambda **kw: True,
        "LoRAConfig": lambda **kw: kw, "WorkerWrapperBase": worker}
    monkeypatch.setenv("RANK", "0")
    exec(compile(ast.Module(body=[method], type_ignores=[]), "worker-method", "exec"), namespace)
    instance = SimpleNamespace(lora_config={"max_loras": 1} if lora_enabled else {},
                               config=SimpleNamespace(dtype="bfloat16", quantization=None))
    namespace["_init_worker"](instance, [{"vllm_config": SimpleNamespace()}])
    assert hijack.call_count == int(lora_enabled)
    worker.return_value.init_worker.assert_called_once()
