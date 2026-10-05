"""Run the production validation method without importing GPU trainer modules."""
import ast
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import uuid

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

from verl import DataProto


@pytest.mark.parametrize("indices", [[0, 1], [0, 0, 0, 1]])
@pytest.mark.parametrize("async_rollout", [False, True])
def test_validation_logs_and_groups_generated_rows(indices, async_rollout):
    source = Path(__file__).resolve().parents[1] / "verl/trainer/ppo/ray_trainer.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    trainer_class = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RayPPOTrainer")
    method = next(n for n in trainer_class.body if isinstance(n, ast.FunctionDef) and n.name == "_validate")
    generated_uids = []
    def generate(batch):
        assert len(batch) == 2
        expanded = batch.select_idxs(indices)
        generated_uids.extend(expanded.non_tensor_batch["uid"])
        expanded.batch["responses"] = torch.full((len(indices), 2), 7)
        expanded.batch["response_mask"] = torch.ones(len(indices), 2)
        return expanded
    def metrics(sources, uids, extra):
        assert list(uids) == generated_uids
        assert len(sources) == len(uids) == len(extra["reward"]) == len(indices)
        assert len(set(uids)) == 2
        return {"bcp": {"reward": {"mean@1": .5}}}
    namespace = {"DataProto": DataProto, "np": np, "uuid": uuid, "defaultdict": defaultdict,
                 "_configured_rollout_workflow": lambda config: None,
                 "_scalarize_reward_extra_info": lambda key, val: None,
                 "compute_reward": lambda batch, fn: (torch.ones(len(batch), 2), {}),
                 "process_validation_metrics": metrics}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    trainer = SimpleNamespace(
        config=OmegaConf.create({"actor_rollout_ref": {"rollout": {"val_kwargs": {
            "n": 1, "temperature": 0, "do_sample": False}}},
            "reward_model": {"enable": False}, "trainer": {"validation_data_dir": "unused"}}),
        val_dataloader=[{"input_ids": torch.tensor([[1, 1], [2, 2]]),
                         "data_source": np.array(["bcp", "bcp"], dtype=object),
                         "reward_model": np.array([{"ground_truth": "A"}, {"ground_truth": "B"}], dtype=object)}],
        tokenizer=SimpleNamespace(eos_token_id=0, pad_token_id=0,
                                  decode=lambda ids, **kwargs: str(ids[0].item())),
        global_steps=2, async_rollout_mode=async_rollout, val_reward_fn=object(),
        actor_rollout_wg=SimpleNamespace(generate_sequences=generate),
        async_rollout_manager=SimpleNamespace(generate_sequences=generate),
        _maybe_log_val_generations=Mock(), _dump_generations=Mock())
    result = namespace["_validate"](trainer)
    assert result["val-core/bcp/reward/mean@1"] == .5
    logged = trainer._dump_generations.call_args.kwargs
    assert logged["inputs"] == [str(i+1) for i in indices]
    assert logged["gts"] == [["A", "B"][i] for i in indices]
    assert logged["outputs"] == ["7"] * len(indices)
    assert logged["trajectory_fields"]["uid"] == generated_uids
