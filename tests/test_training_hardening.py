import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer.ppo.dynamic_groups import informative_groups, sample_groups
from verl.trainer.ppo.core_algos import compute_graphrpo_advantage, compute_policy_loss_graphrpo
from verl.utils.padding_trim import trim_micro_batch


def config(**kw):
    return OmegaConf.create(dict(adv_estimator='graphrpo', graphrpo_dynamic_sampling=True,
                                 graphrpo_dynamic_max_gen_batches=3, graphrpo_dynamic_min_groups=1, **kw))


def batch(rewards=(0, 0, 1), uid='q'):
    # Two streams for the first episode must survive selection together.
    return DataProto.from_dict(tensors={'rm_scores': torch.tensor([[r, 0.] for r in rewards]),
        'response_mask': torch.ones(3, 2), 'graph_edit_credit_mask': torch.zeros(3, 2),
        'process_reward_mask': torch.zeros(3, 2)}, non_tensors={
        'uid': np.array([uid] * 3), 'gen_uid': np.array([uid+'a', uid+'a', uid+'b'])})


def test_dynamic_deduplicates_and_retains_whole_group():
    b = batch()
    assert informative_groups(b, config())[0] == [[0, 1, 2]]
    b.batch['rm_scores'].zero_()
    assert informative_groups(b, config())[0] == []
    b.batch['process_reward_mask'][1, 0] = -.3
    assert informative_groups(b, config())[0] == [[0, 1, 2]]
    assert informative_groups(b, config(), 'E')[0] == []
    assert informative_groups(b, config(), 'M')[0] == []
    b.batch['graph_edit_credit_mask'][2, 1] = 1
    assert informative_groups(b, config(graphrpo_memory_only=True))[0] == [[0, 1, 2]]
    b.batch['response_mask'][:, 1] = 0
    assert informative_groups(b, config(), 'M')[0] == []


def test_dynamic_bounded_exhaustion_disabled_and_partial_groups():
    c = config()
    generated = []
    def generate(value):
        generated.append(value)
        return batch((0, 0, int(value == 2)), str(value))
    following, metrics = iter([1, 2, 3]), {}
    result = sample_groups(0, following, generate, c, 'E', 2, metrics)
    assert generated == [0, 1, 2] and next(following) == 3
    assert result.non_tensor_batch['uid'].tolist() == ['2'] * 3
    assert metrics['graphrpo/dynamic_gen_batches'] == 3
    empty = sample_groups(0, iter([]), generate, c, 'E', 2, {})
    assert empty.meta_info['dynamic_empty'] and not empty.batch['response_mask'].any()
    c.graphrpo_dynamic_sampling = False
    untouched = sample_groups(0, iter([2]), generate, c, 'E', 2, {})
    assert untouched.batch['response_mask'].all()


def test_dynamic_multiple_batches_concatenate_without_candidate_mixing():
    c = config()
    c.graphrpo_dynamic_min_groups = 2
    result = sample_groups('x', iter(['y']), lambda x: batch(uid=x), c, 'E', 2, {})
    assert result.non_tensor_batch['gen_uid'].tolist() == ['xa', 'xa', 'xb', 'ya', 'ya', 'yb']
    broken = batch()
    broken.batch['rm_scores'][1, 0] = 1
    with pytest.raises(ValueError, match='Streams disagree'):
        informative_groups(broken, c)


def test_foldgrpo_filters_outcome_ties_even_with_process_labels():
    c = config()
    c.adv_estimator = 'foldgrpo'
    b = batch((1,1,1))
    b.batch['process_reward_mask'].fill_(-1)
    assert informative_groups(b,c)[0] == []


def test_content_filter_400_is_recoverable_but_other_400_is_not():
    import httpx
    from agents.graph_rpo_continuation import recoverable
    for code, expected in [('content_filter',True),('bad_parameter',False)]:
        response = httpx.Response(400, json={'error':{'code':code}},
            request=httpx.Request('POST','https://example.invalid'))
        error = httpx.HTTPStatusError('rejected',request=response.request,response=response)
        assert recoverable(error) == expected


def test_driver_progress_restores_epoch_independent_of_optimizer_steps(tmp_path):
    import ast
    import os
    from pathlib import Path
    tree = ast.parse(Path('verl/trainer/ppo/ray_trainer.py').read_text(encoding='utf8'))
    cls = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='RayPPOTrainer')
    cls.body = [n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in {'_save_checkpoint','_load_checkpoint'}]
    namespace = dict(os=os,torch=torch,DataProto=DataProto,Role=SimpleNamespace(Critic='critic'),
        find_latest_ckpt_path=lambda root:str(Path(root)/'global_step_7'))
    exec(compile(ast.Module(body=[cls],type_ignores=[]),'<driver>', 'exec'),namespace)
    trainer = namespace['RayPPOTrainer'].__new__(namespace['RayPPOTrainer'])
    trainer.config = OmegaConf.create({'trainer':{'default_local_dir':str(tmp_path),'default_hdfs_dir':None,
        'resume_mode':'auto','del_local_ckpt_after_load':False},
        'actor_rollout_ref':{'actor':{'checkpoint':{'async_save':False}}},
        'algorithm':{'graphrpo_dynamic_sampling':True}})
    trainer.use_critic = False
    trainer.global_steps, trainer._training_epoch = 7, 3
    trainer.actor_rollout_wg = SimpleNamespace(save_checkpoint=lambda *a,**k:None,load_checkpoint=lambda *a,**k:None)
    restored = []
    trainer.train_dataloader = SimpleNamespace(state_dict=lambda:{'consumed':19},load_state_dict=restored.append)
    trainer._save_checkpoint()
    trainer.global_steps, trainer._training_epoch = 0, 0
    trainer._load_checkpoint()
    assert (trainer.global_steps,trainer._training_epoch) == (7,3)
    assert restored == [{'consumed':19}]
    (tmp_path/'global_step_7'/'data.pt').unlink()
    with pytest.raises(ValueError, match='dataloader state'):
        trainer._load_checkpoint()


def test_decision_scaling_cap_and_rejected_credit():
    mask = torch.ones(2, 10)
    decisions = torch.zeros_like(mask)
    decisions[:, 2] = 1
    credit = decisions.clone()
    mask[:, 9] = 0
    credit[:, 9] = float('nan')  # rejected tokens must not poison active credit
    advantage, _ = compute_graphrpo_advantage(torch.zeros_like(mask), mask,
        np.array(['q', 'q']), np.array(['a', 'b']), graph_edit_credit_mask=credit,
        graph_decision_mask=decisions, config={'graphrpo_normalize_decision_tokens': True,
        'graphrpo_decision_scale_max': 3.0})
    assert advantage[:, 2].tolist() == [3., 3.]
    assert torch.isfinite(advantage).all() and not advantage[:, 9].any()
    credit[0,2] = float('nan')
    with pytest.raises(ValueError, match='Nonfinite'):
        compute_graphrpo_advantage(torch.zeros_like(mask),mask,np.array(['q','q']),np.array(['a','b']),
            graph_edit_credit_mask=credit,config={})


@pytest.mark.parametrize('bad_field', ['log_prob', 'advantages', 'rollout_is_weights'])
def test_nonfinite_masked_tokens_have_finite_zero_gradient(bad_field):
    values = dict(old_log_prob=torch.zeros(1, 2), log_prob=torch.zeros(1, 2),
                  advantages=torch.ones(1, 2), rollout_is_weights=torch.ones(1, 2))
    values[bad_field][0, 1] = float('nan')
    values['log_prob'].requires_grad_()
    actor = SimpleNamespace(clip_ratio=.2, clip_ratio_low=None, clip_ratio_high=None, global_batch_info={})
    mask = torch.tensor([[1., 0.]])
    loss, metrics = compute_policy_loss_graphrpo(**values, response_mask=mask,
        graphrpo_loss_weights=mask, config=actor)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(values['log_prob'].grad).all()
    assert values['log_prob'].grad[0, 1] == 0 and metrics['actor/pg_nonfinite_tokens'] == 0
    loss, metrics = compute_policy_loss_graphrpo(**values, response_mask=torch.ones_like(mask),
        graphrpo_loss_weights=torch.ones_like(mask), config=actor)
    assert not torch.isfinite(loss) and metrics['actor/pg_nonfinite_tokens'] == 1


def test_padding_trim_preserves_active_logits_and_positions():
    mask = torch.tensor([[0]*5 + [1]*5 + [0]*6, [0]*4 + [1]*8 + [0]*4])
    tokens = torch.arange(32).reshape(2, 16)
    b = dict(input_ids=tokens, attention_mask=mask, position_ids=mask.cumsum(-1), responses=tokens[:, 8:])
    trimmed, width = trim_micro_batch(b, 4)
    assert width == 8 and trimmed['input_ids'].shape[-1] == 8
    # A tiny causal model whose states depend on every preceding visible token.
    def logits(x):
        return (x['input_ids'] * x['attention_mask']).cumsum(-1) + x['position_ids']
    original = logits(b)[:, -9:-1]
    shorter = logits(trimmed)[:, -5:-1]
    assert torch.equal(original[:, :4], shorter)
    assert trimmed['responses'].tolist() == b['responses'][:, :4].tolist()
    assert trim_micro_batch(b, 0)[0] is b
    mm = {**b, 'multi_modal_inputs': {}}
    assert trim_micro_batch(mm, 4)[0] is mm


def test_fragment_cache_release_is_opt_in(monkeypatch):
    from verl.utils.padding_trim import release_fragmented_cache
    calls = []
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'memory_reserved',lambda:30 * 1024**3)
    monkeypatch.setattr(torch.cuda,'memory_allocated',lambda:4 * 1024**3)
    monkeypatch.setattr(torch.cuda,'empty_cache',lambda:calls.append(1))
    monkeypatch.delenv('VERL_FRAG_EMPTY_CACHE_GB',raising=False)
    release_fragmented_cache()
    assert not calls
    monkeypatch.setenv('VERL_FRAG_EMPTY_CACHE_GB','24')
    release_fragmented_cache()
    assert calls == [1]


@pytest.mark.parametrize('policy,budget,service,accepted', [(True,True,False,True),
    (False,True,False,False), (True,False,False,False), (True,True,True,False)])
def test_policy_budget_failure_never_converts_service_errors(policy,budget,service,accepted):
    from agents.graph_rpo_continuation import _checked_episode, ContinuationUnavailable
    output = SimpleNamespace(reward_score=1., extra_fields={'agent_name':'main','env_stats':{},
        'hit_timeout':True,'policy_time_budget_exhausted':policy})
    async def run(*a, **kw):
        return [output]
    ctx = SimpleNamespace(config=SimpleNamespace(actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(
        plugin={'graph_rpo_timeout_as_failure':budget}))))
    session = SimpleNamespace(error=None, env=SimpleNamespace(env_fail=service))
    if accepted:
        assert asyncio.run(_checked_episode(None,ctx,run,session))[0].reward_score == 0
    else:
        with pytest.raises(ContinuationUnavailable):
            asyncio.run(_checked_episode(None,ctx,run,session))
