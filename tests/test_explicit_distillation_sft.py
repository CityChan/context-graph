import asyncio
import copy
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf
import pandas as pd
import pytest

from agents.exact_context import checksum, tokenizer_signature, validate_agent_context
from agents.sft_teacher_guidance import DISTILL_GUIDANCE, DISTILL_PROFILE
from scripts import build_explicit_distillation_sft as builder, generate_bcp_graph_sft as generator
from scripts.check_explicit_distillation_sft import check
from scripts.route_a_client import RouteAClient, execution_text
from tests.test_auto_graph_sft import arguments
from tests.test_bcp_graph_sft import fixture, row
from tests.test_route_a_sft import Tokenizer, record


def configuration():
    args = arguments()
    args.collection_mode = 'explicit_distillation'
    args.graph_protocol, args.transport = 'explicit_tools', 'vllm_tokens'
    args.teacher_profile, args.seed = DISTILL_PROFILE, 42
    return args, generator.make_config(args)


def native_fixture(query='Find evidence'):
    episode, trajectory, requests = fixture()
    tokenizer = Tokenizer()
    for r in requests:
        r['messages'] = [{'role': 'system', 'content': 'Permanent JSON interface\n\n' + DISTILL_GUIDANCE},
                         {'role': 'user', 'content': query + '\nGraph: n2 obsolete evidence'}]
        r.update(record(r['messages'], r['response']))
        r['output_ids'] = tokenizer.encode('<think>Check the evidence.</think>\n' + r['response']) + [0]
        r.update(native_token_ids=True, requested_model='test-teacher', returned_model='test-teacher')
    return episode, trajectory, requests


def collection(tmp_path, name='run'):
    root = tmp_path / name
    root.mkdir()
    data, heldout = tmp_path / 'train.parquet', tmp_path / 'heldout.parquet'
    if not data.exists():
        pd.DataFrame([row('Find evidence'), row('Another question')]).to_parquet(data)
        pd.DataFrame([row('Independent test question', 'test')]).to_parquet(heldout)
    args, config = configuration()
    manifest = dict(profile=generator.DISTILL_PROFILE, collection_guidance=DISTILL_GUIDANCE,
                    tokenizer_sha256=tokenizer_signature(Tokenizer()), settings=vars(args), indices=[0, 1],
                    source_sha256=builder.file_hash(data), heldout_sha256=builder.file_hash(heldout),
                    config=OmegaConf.to_container(config, resolve=True))
    (root / 'manifest.json').write_text(json.dumps(manifest))
    for index, query in enumerate(('Find evidence', 'Another question')):
        directory = root / 'episodes' / str(index)
        directory.mkdir(parents=True)
        ep, trajectory, requests = native_fixture(query)
        ep.update(source_index=index, candidate=0, benchmark='bcp', question_hash=checksum(query.casefold()),
                  trajectory=f'episodes/{index}/trajectory.json', requests=f'episodes/{index}/requests.jsonl')
        (directory / 'episode.json').write_text(json.dumps(ep))
        (directory / 'trajectory.json').write_text(json.dumps(trajectory))
        (directory / 'requests.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in requests))
    return root, SimpleNamespace(runs=[str(root)], data=str(data), heldout=str(heldout),
                                 output=str(tmp_path / 'export'), tokenizer='student', max_length=65536,
                                 seed=42, validation_fraction=0.5, control_ratio=0.5, smoke=False, require_complete=True)


def test_guidance_transform_keeps_task_evidence_and_original_objects():
    _, _, requests = native_fixture()
    before = copy.deepcopy(requests[0]['messages'])
    cleaned, removed = builder.strip_guidance(before)
    assert removed == 1 and cleaned[0]['content'] == 'Permanent JSON interface'
    assert cleaned[1] == before[1] and DISTILL_GUIDANCE in before[0]['content']
    assert builder.strip_guidance(cleaned) == (cleaned, 0)
    before[0]['content'] = before[0]['content'].replace('Complete the task', 'Altered instructions')
    with pytest.raises(ValueError, match='registered_system_block'):
        builder.strip_guidance(before)


def test_native_reasoning_and_eos_are_labels_with_student_prefix():
    ep, trajectory, requests = native_fixture()
    reason, audit = builder.audit_episode(ep, trajectory, requests)
    assert reason is None
    transformed, provenance = builder.prepare_requests(requests, audit['events'], Tokenizer(), 'test-teacher')
    assert transformed[0]['output_ids'] == requests[0]['output_ids']
    assert all(p['input_changed'] and p['removed_guidance_blocks'] == 1 for p in provenance)
    assert DISTILL_GUIDANCE not in str(transformed[0]['messages'])
    for segment in builder.segments_from_requests(transformed, 65536, tokenizer_signature(Tokenizer())):
        ids, mask = validate_agent_context(segment, 65536, tokenizer_signature(Tokenizer()))
        target = [i for i, train in zip(ids, mask) if train]
        assert '<think>Check the evidence.</think>' in Tokenizer().decode(target)
        assert target[-1] == 0


@pytest.mark.parametrize('problem', ['missing_node', 'trace_argument', 'output_ids', 'model'])
def test_reject_untraceable_or_invisible_targets(problem):
    ep, trajectory, requests = native_fixture()
    _, audit = builder.audit_episode(ep, trajectory, requests)
    if problem == 'missing_node':
        requests[0]['messages'][1]['content'] = 'No graph references'
    elif problem == 'trace_argument':
        audit['events'][0]['args']['node_id'] = 'n99'
    elif problem == 'output_ids':
        requests[0]['output_ids'] = Tokenizer().encode('{}')
    else:
        requests[0]['returned_model'] = 'different-teacher'
    with pytest.raises(ValueError):
        builder.prepare_requests(requests, audit['events'], Tokenizer(), 'test-teacher')


def test_build_deduplicates_runs_and_checks_actual_training_files(tmp_path, monkeypatch):
    root, args = collection(tmp_path)
    other, _ = collection(tmp_path, 'second')
    args.runs.append(str(other))
    monkeypatch.setattr(builder, 'split_for', lambda q, *a: 'train' if q == checksum('find evidence') else 'validation')
    report = builder.build(args, Tokenizer())
    assert report['outputs']['train'] == report['outputs']['validation'] == 2
    assert report['counts']['duplicate_segments'] == 4
    assert report['loaded_coverage']['train']['operation_calls']['prune'] == 1
    assert report['loaded_coverage']['train']['prune_episodes'] == 1
    assert check(args.output, args.heldout, Tokenizer(), 65536)['passed']
    path = Path(args.output) / 'train.parquet'
    from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
    dataset = MultiTurnSFTDataset(str(path), Tokenizer(), config={'max_length': 65536, 'truncation': 'error', 'loss_mask_mode': 'agent_context'})
    assert dataset[0]['loss_mask'].sum() > 0
    dataset.loss_mask_mode = 'chatml'
    with pytest.raises(ValueError, match='requires loss_mask_mode'):
        dataset[0]
    path.write_bytes(path.read_bytes() + b'tamper')
    with pytest.raises(ValueError, match='export_file_changed'):
        check(args.output, args.heldout, Tokenizer(), 65536)


def test_overlap_rejected_before_success_filter(tmp_path):
    root, args = collection(tmp_path)
    pd.DataFrame([row(' FIND  EVIDENCE ', 'test')]).to_parquet(args.heldout)
    with pytest.raises(ValueError, match='overlap'):
        builder.build(args, Tokenizer())


def test_manifest_wrong_protocol_and_changed_files_fail(tmp_path):
    root, args = collection(tmp_path)
    manifest = json.loads((root / 'manifest.json').read_text())
    manifest['config']['actor_rollout_ref']['rollout']['plugin']['structured_graph_controller'] = True
    with pytest.raises(ValueError, match='scheduled_graph_controller'):
        builder.validate_manifest(manifest, args.data, args.heldout, tokenizer_signature(Tokenizer()))
    with pytest.raises(ValueError, match='tokenizer_mismatch'):
        builder.validate_manifest(manifest, args.data, args.heldout, 'wrong')
    pd.DataFrame([row('changed source')]).to_parquet(args.data)
    with pytest.raises(ValueError, match='file_changed'):
        builder.build(args, Tokenizer())


def test_explicit_thinking_transport_records_schema_but_does_not_constrain_reasoning(tmp_path):
    args, config = configuration()
    sent = []
    tokenizer = Tokenizer()
    text = '<think>Inspect memory.</think>\n{"tool":"graph_read","arguments":{}}'
    async def post(url, json):
        sent.append(json)
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {'model': 'test-teacher', 'choices': [
            {'token_ids': tokenizer.encode(text) + [0], 'finish_reason': 'stop'}]})
    client = RouteAClient(args, tokenizer, config.actor_rollout_ref.rollout, tmp_path / 'requests.jsonl', client=SimpleNamespace(post=post))
    result = asyncio.run(client.create_completion([1, 2], messages=[{'role': 'user', 'content': 'Inspect'}],
                                                  max_new_tokens=256, structured_outputs={'json': {'type': 'object'}}))
    assert 'structured_outputs' not in sent[0]
    assert result['choices'][0]['message']['content'] == '{"tool":"graph_read","arguments":{}}'
    saved = json.loads((tmp_path / 'requests.jsonl').read_text())
    assert saved['schema'] and saved['output_ids'] == tokenizer.encode(text) + [0]
    assert execution_text(tokenizer, saved['output_ids'], explicit=False) == text


def test_thinking_interface_opt_in_and_controller_profile_unchanged():
    args, config = configuration()
    from agents.utils import Agent
    from agents.action_contract import ActionContract, PROMPT, THINKING_PROMPT
    agent = Agent(None, [{'role': 'system', 'content': 'System'}, {'role': 'user', 'content': 'Task'}],
                  Tokenizer(), config.actor_rollout_ref.rollout, action_contract=ActionContract('LocalSearch', None))
    assert agent.chat_template_kwargs['enable_thinking'] is True
    assert 'or a thinking block' not in THINKING_PROMPT and 'or a thinking block' in PROMPT
    args = arguments()
    args.collection_mode, args.transport = 'route_a', 'vllm_tokens'
    plugin = generator.make_config(args).actor_rollout_ref.rollout.plugin
    assert plugin.structured_graph_controller and not plugin.get('explicit_action_thinking', False)


def test_native_explicit_collection_through_agent_and_export_audit(tmp_path, monkeypatch):
    from agents import graph_agent_isolated
    from scripts import route_a_client

    class Env:
        def __init__(self, *args):
            self.stats = Counter()
            self.instance_info = {'problem_statement': 'Find temperature'}
            self.is_finish = self.env_fail = False
        async def init_env(self, item):
            pass
        async def run_action(self, response):
            self.stats['action'] += 1
            if 'function=finish' in response:
                self.is_finish = True
                return {'action': 'finish'}
            return {'observation': 'Temporary search result: unneeded page'}
        async def get_reward(self, *args):
            return '', 1.0, {}
        def close(self):
            pass

    tokenizer = Tokenizer()
    actions = iter([{'tool': 'search', 'arguments': {'query': 'temperature'}},
                    {'tool': 'prune', 'arguments': {'node_id': 'n2'}},
                    {'tool': 'finish', 'arguments': {'answer': '20 C'}}])
    sent = []
    async def post(url, json):
        sent.append(json)
        ids = tokenizer.encode('<think>Check observed evidence.</think>\n' + __import__('json').dumps(next(actions))) + [0]
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {'model': 'test-teacher', 'choices': [
            {'token_ids': ids, 'finish_reason': 'stop'}]})
    async def close():
        pass
    async def preflight(*args):
        pass
    monkeypatch.setattr(graph_agent_isolated, 'select_env', lambda *a: Env)
    monkeypatch.setattr(route_a_client, 'RouteAClient', lambda *a: RouteAClient(*a, client=SimpleNamespace(post=post, aclose=close)))
    monkeypatch.setattr(generator, 'preflight', preflight)
    monkeypatch.setenv('AGENT_BUDGET_MODE', 'working_context')
    args, config = configuration()
    asyncio.run(generator.collect(args, [row('Find temperature')], [0], config, tokenizer, tmp_path))
    episode = builder.read_json(tmp_path / 'episodes/000000-00/episode.json')
    trajectory = builder.read_json(tmp_path / episode['trajectory'])
    requests = builder.read_jsonl(tmp_path / episode['requests'])
    reason, audit = builder.audit_episode(episode, trajectory, requests)
    assert reason is None
    assert len(requests) == 3 and all('structured_outputs' not in r for r in sent)
    transformed, provenance = builder.prepare_requests(requests, audit['events'], tokenizer, args.teacher_model)
    assert sum(p['removed_guidance_blocks'] for p in provenance) == 3
    assert all('Check observed evidence.' in tokenizer.decode(r['output_ids']) for r in transformed)


def test_export_rejects_incomplete_and_overlength_without_truncation(tmp_path, monkeypatch):
    root, args = collection(tmp_path)
    args.max_length = 1
    args.smoke = True
    with pytest.raises(ValueError, match='No accepted'):
        builder.build(args, Tokenizer())
    report = builder.read_json(Path(args.output) / 'manifest.json')
    assert report['outputs'] == {}
    assert any('truncation' in key for key in report['counts'])


@pytest.mark.parametrize('finish_reason', ['stop', 'length'])
def test_preflight_checks_the_thinking_path_and_rejects_output_limit(tmp_path, monkeypatch, finish_reason):
    args, config = configuration()
    args.benchmark = 'scienceworld'
    calls = []
    class Teacher:
        async def create_completion(self, ids, **kwargs):
            calls.append(kwargs)
            return {'choices': [{'message': {'content': '{"tool":"graph_read","arguments":{}}'}, 'finish_reason': finish_reason}]}
        async def close(self):
            pass
    class ThinkingTokenizer(Tokenizer):
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs['enable_thinking'] is True and kwargs['return_dict'] is False
            return super().apply_chat_template(messages, **kwargs)
    monkeypatch.setattr(generator, 'collection_client', lambda *a: Teacher())
    if finish_reason == 'length':
        with pytest.raises(ValueError, match='turn budget'):
            asyncio.run(generator.preflight(args, ThinkingTokenizer(), config.actor_rollout_ref.rollout, tmp_path))
    else:
        asyncio.run(generator.preflight(args, ThinkingTokenizer(), config.actor_rollout_ref.rollout, tmp_path))
        assert len(calls) == 2
    assert all(c['max_new_tokens'] == args.turn_max_tokens for c in calls)


def test_training_preflight_refuses_missing_episodes(tmp_path, monkeypatch):
    root, args = collection(tmp_path)
    (root / 'episodes/1/episode.json').unlink()
    monkeypatch.setattr(builder, 'split_for', lambda *a: 'train')
    with pytest.raises(ValueError, match='collection_incomplete'):
        builder.build(args, Tokenizer())
    with pytest.raises(ValueError, match='collection_incomplete'):
        check(args.output, args.heldout, Tokenizer(), 65536)


def test_launchers_are_syntactically_valid_and_default_to_64k():
    import subprocess
    for name in ('generate_explicit_distillation_sft_4node_idev.sh', 'train_explicit_distillation_sft_4node_idev.sh',
                 'generate_route_a_sft_qwen35_9b_4node_idev.sh', 'smoke_train_contextgraph_sft_qwen36_27b_4node_idev.sh'):
        subprocess.run(['bash', '-n', str(Path('scripts') / name)], check=True)
    assert '${CONTEXT_LENGTH:-65536}' in Path('scripts/generate_explicit_distillation_sft_4node_idev.sh').read_text()
    assert '${MAX_LENGTH:-65536}' in Path('scripts/train_explicit_distillation_sft_4node_idev.sh').read_text()
