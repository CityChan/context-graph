"""CPU/Gloo regression for the actor's real update loop (not a GPU FSDP smoke)."""

import ast
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from verl.utils.distributed_loss import require_finite_losses


ROOT = Path(__file__).resolve().parents[1]


def _actor_class():
    # Avoid optional GPU engine imports; execute the actual update_policy body.
    source = ROOT / 'verl/workers/actor/dp_actor.py'
    tree = ast.parse(source.read_text(encoding='utf8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DataParallelPPOActor')
    cls.bases = []
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'update_policy']
    cls.body[0].decorator_list = []
    from verl import DataProto
    from verl.utils.py_functional import append_to_dict
    from verl.trainer.ppo.core_algos import agg_loss
    namespace = dict(torch=torch, DataProto=DataProto, get_device_id=lambda: 'cpu',
                     require_finite_losses=require_finite_losses, append_to_dict=append_to_dict,
                     agg_loss=agg_loss, kl_penalty=lambda logprob, ref_logprob, **kw: logprob - ref_logprob,
                     get_policy_loss_fn=lambda mode: lambda **kw: (
                         (kw['log_prob'] * kw['advantages'] * kw['response_mask']).sum(), {}))
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace[cls.name]


def _worker(rank, rendezvous, output):
    from datetime import timedelta
    from torch.nn.parallel import DistributedDataParallel
    from verl import DataProto

    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=30))
    rows = []
    try:
        for stage in ('pg', 'kl', 'total', 'empty_rank', 'finite'):
            module = torch.nn.Linear(1, 1, bias=False)
            torch.nn.init.ones_(module.weight)
            actor = _actor_class()()
            actor.actor_module = DistributedDataParallel(module)
            actor.actor_optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
            actor.scaler = None
            actor.ulysses_sequence_parallel_size = 1
            actor.config = SimpleNamespace(
                policy_loss={'loss_mode': 'graphrpo'}, ppo_epochs=1, use_dynamic_bsz=False,
                ppo_mini_batch_size=2, ppo_micro_batch_size_per_gpu=1, entropy_coeff=1 if stage == 'total' else 0,
                calculate_entropy=stage == 'total', loss_agg_mode='token-mean', use_kl_loss=stage == 'kl',
                kl_loss_type='kl', kl_loss_coef=1., global_batch_info={})
            counts = {'forward': 0, 'backward': 0, 'step': 0}
            def backward_hook(grad):
                counts['backward'] += 1
                return grad
            module.weight.register_hook(backward_hook)
            def forward(inputs, **kw):
                counts['forward'] += 1
                value = actor.actor_module(torch.ones(1, 1))
                entropy = value if stage == 'total' else None
                if stage == 'total' and rank == 0 and counts['forward'] == 2:
                    entropy = value * float('nan')
                return entropy, value
            def step():
                counts['step'] += 1
                norm = torch.nn.utils.clip_grad_norm_(module.parameters(), 100.)
                actor.actor_optimizer.step()
                return norm
            actor._forward_micro_batch = forward
            actor._optimizer_step = step
            tensors = {key: torch.ones(2, 1) for key in (
                'responses', 'response_mask', 'input_ids', 'attention_mask', 'position_ids',
                'old_log_probs', 'advantages', 'graphrpo_loss_weights')}
            tensors['ref_log_prob'] = torch.zeros(2, 1)
            if rank == 0 and stage == 'pg':
                tensors['advantages'][1] = float('nan')
            if rank == 0 and stage == 'kl':
                tensors['ref_log_prob'][1] = float('inf')
            if rank == 0 and stage == 'empty_rank':
                tensors['response_mask'].zero_()
                tensors['graphrpo_loss_weights'].zero_()
            data = DataProto.from_dict(tensors=tensors, meta_info={'temperature': 1.})
            error = None
            try:
                actor.update_policy(data)
            except FloatingPointError as exc:
                error = str(exc)
            rows.append(dict(stage=stage, error=error, weight=module.weight.item(),
                             grad_cleared=module.weight.grad is None, **counts))
            dist.barrier()
        Path(output).write_text(json.dumps(rows), encoding='utf8')
    finally:
        dist.destroy_process_group()


def test_single_process_finite_guard():
    require_finite_losses(pg_loss=torch.tensor(0.))
    with pytest.raises(FloatingPointError, match='kl_loss'):
        require_finite_losses(pg_loss=torch.tensor(1.), kl_loss=torch.tensor(float('nan')))


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason='Gloo unavailable')
def test_two_ranks_agree_on_nonfinite_loss_and_empty_rank_participates(tmp_path):
    rendezvous = (tmp_path / 'rendezvous').as_uri()
    env = {**os.environ, 'PYTHONPATH': str(ROOT)}
    processes, logs = [], []
    try:
        for rank in range(2):
            log = (tmp_path / f'{rank}.log').open('w', encoding='utf8')
            logs.append(log)
            processes.append(subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), str(rank), rendezvous, str(tmp_path / f'{rank}.json')],
                stdout=log, stderr=subprocess.STDOUT, env=env, cwd=ROOT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0))
        for proc in processes:
            proc.wait(timeout=90)
    finally:
        for proc in processes:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
        for log in logs:
            log.close()
    assert all(proc.returncode == 0 for proc in processes), '\n'.join(
        (tmp_path / f'{rank}.log').read_text(encoding='utf8') for rank in range(2))
    ranks = [json.loads((tmp_path / f'{rank}.json').read_text()) for rank in range(2)]
    for left, right in zip(*ranks):
        assert left['error'] == right['error']
        for row in (left, right):
            assert row['grad_cleared']
            if row['stage'] in {'pg', 'kl', 'total'}:
                assert row['error'] and row['step'] == 0 and row['backward'] == 1
                component = {'pg': 'pg_loss', 'kl': 'kl_loss', 'total': 'total_loss'}[row['stage']]
                assert component in row['error']
                assert row['weight'] == 1.
            else:
                assert row['error'] is None and row['step'] == 1 and row['backward'] == 2
                assert row['weight'] < 1.
        assert left['weight'] == right['weight']


if __name__ == '__main__':
    _worker(int(sys.argv[1]), sys.argv[2], sys.argv[3])
