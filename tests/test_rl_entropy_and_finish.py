"""Regression coverage for padded RL entropy and terminal-action penalties."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.utils.checkpoint

from verl.utils.torch_functional import (
    entropy_from_logits,
    entropy_from_logits_with_chunking,
    logprobs_from_logits,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("shape", [(9, 17), (2, 9, 17)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_chunked_entropy_values_gradients_and_bounded_rows(shape, dtype, monkeypatch):
    torch.manual_seed(7)
    raw = torch.randn(shape, dtype=dtype, requires_grad=True)
    logits = raw[..., 1:-1, :]  # Noncontiguous padded response slice for batch > 1.
    expected = entropy_from_logits(logits.float())
    expected_grad = torch.autograd.grad(expected.sum(), raw)[0]
    calls = []
    original = torch.nn.functional.softmax

    def bounded_softmax(chunk, *args, **kwargs):
        calls.append(chunk.shape)
        assert chunk.ndim == 2 and chunk.shape[0] <= 3
        return original(chunk, *args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "softmax", bounded_softmax)
    actual = entropy_from_logits_with_chunking(logits, chunk_size=3)
    actual_grad = torch.autograd.grad(actual.sum(), raw)[0]
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_grad, expected_grad)
    assert len(calls) == (3 if len(shape) == 2 else 6)


def _load_actor_forward():
    # Run the actual method with a tiny CPU model; no Ray/FSDP/GPU initialization.
    path = ROOT / "verl/workers/actor/dp_actor.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_forward_micro_batch")
    namespace = {"torch": torch, "logprobs_from_logits": logprobs_from_logits}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[method.name]


@pytest.mark.parametrize("checkpointing", [False, True])
@pytest.mark.parametrize("grad_enabled", [False, True])
def test_padded_actor_honors_selected_entropy_function(checkpointing, grad_enabled):
    torch.manual_seed(11)
    logits = torch.randn(2, 9, 17, requires_grad=True)
    calls = []

    def selected_entropy(value):
        calls.append(value.shape)
        return entropy_from_logits_with_chunking(value, chunk_size=3)

    actor = SimpleNamespace(
        device_name="cpu", param_dtype=torch.bfloat16,
        use_remove_padding=False, use_fused_kernels=False,
        config=SimpleNamespace(entropy_checkpointing=checkpointing),
        actor_module=lambda **kwargs: SimpleNamespace(logits=logits * 1.0),
        compute_entropy_from_logits=selected_entropy,
    )
    batch = {
        "input_ids": torch.zeros(2, 9, dtype=torch.long),
        "attention_mask": torch.ones(2, 9),
        "position_ids": torch.arange(9).expand(2, -1),
        "responses": torch.zeros(2, 5, dtype=torch.long),
    }
    with torch.set_grad_enabled(grad_enabled):
        entropy, log_probs = _load_actor_forward()(actor, batch, 1.0, calculate_entropy=True)
    expected = entropy_from_logits(logits[:, -6:-1, :])
    torch.testing.assert_close(entropy, expected)
    if grad_enabled:
        entropy.sum().backward()
        expected_grad = torch.autograd.grad(expected.sum(), logits)[0]
        torch.testing.assert_close(logits.grad, expected_grad)
    else:
        assert not entropy.requires_grad
    assert calls and all(shape == (2, 5, 17) for shape in calls)
    assert log_probs.shape == (2, 5)


@pytest.mark.parametrize("module", ["fold_agent.py", "graph_agent_isolated.py"])
def test_unfolded_trajectory_does_not_penalize_finish(module):
    path = ROOT / "agents" / module
    tree = ast.parse(path.read_text(encoding="utf-8"))
    block = next(n for n in ast.walk(tree) if isinstance(n, ast.If) and ast.unparse(n.test) == "rollout_status['unfolded_main']")
    messages = [
        {"role": "system", "content": "task"},
        {"role": "assistant", "content": "<function=search>query</function>"},
        {"role": "assistant", "content": "<function=branch>subtask</function>"},
        {"role": "assistant", "content": "<function=finish>answer</function>"},
        {"role": "assistant", "content": "<function=select>n1</function>"},
    ]
    assigned = {}
    agent = SimpleNamespace(
        messages=lambda: messages,
        set_process_reward=lambda indices, value: assigned.update({i: value for i in indices}),
    )
    namespace = {
        "agent": {"main": agent}, "rollout_status": {"unfolded_main": 1},
        "GRAPH_OP_MARKERS": ("<function=select>",),
    }
    exec(compile(ast.Module(body=[block], type_ignores=[]), str(path), "exec"), namespace)
    assert 3 not in assigned  # A successful terminal action stays exempt.
    assert 2 not in assigned  # Branch exemption is retained.
    assert assigned[1] == -1  # Existing research-turn shaping is unchanged.
    if module == "graph_agent_isolated.py":
        assert 4 not in assigned


def test_qwen_rl_enables_chunked_checkpointed_entropy():
    source = (ROOT / "scripts/train_bcp_qwen35_9b_50step.sh").read_text(encoding="utf-8")
    assert "actor_rollout_ref.actor.entropy_from_logits_with_chunking=True" in source
    assert "actor_rollout_ref.actor.entropy_checkpointing=True" in source
