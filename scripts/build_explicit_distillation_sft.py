"""Explicit-policy context distillation: audited native targets, transformed prefixes.

Only the registered system guidance is removed. No environment evidence, graph
state, task instructions or sampled responses are rewritten. Not exact replay.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import copy
import hashlib
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.exact_context import checksum, segments_from_requests, token_list, tokenizer_signature, validate_agent_context
from agents.sft_teacher_guidance import DISTILL_GUIDANCE, DISTILL_START, DISTILL_END
from scripts.build_bcp_graph_sft import audit_episode, read_json, read_jsonl, split_for, GRAPH_TOOLS
from scripts.build_route_a_sft import mix_training
from scripts.generate_bcp_graph_sft import DISTILL_PROFILE, question, select_training_rows
from scripts.route_a_client import execution_text

PROFILE = 'explicit_context_distillation_v1'


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def strip_guidance(messages):
    cleaned = copy.deepcopy(messages)
    removed = 0
    for message in cleaned:
        text = message.get('content')
        if not isinstance(text, str):
            raise ValueError('text_messages_required')
        if DISTILL_START not in text and DISTILL_END not in text:
            continue
        block = '\n\n' + DISTILL_GUIDANCE
        if message.get('role') != 'system' or text.count(block) != 1:
            raise ValueError('guidance_not_registered_system_block')
        message['content'] = text.replace(block, '', 1)
        if DISTILL_START in message['content'] or DISTILL_END in message['content']:
            raise ValueError('ambiguous_guidance_block')
        removed += 1
    return cleaned, removed


def referenced_nodes(payload):
    args = payload['arguments']
    result = list(args.get('node_ids', []))
    result += [args[k] for k in ('node_id', 'source', 'target') if args.get(k)]
    return result


def prepare_requests(requests, events, tokenizer, model):
    """Audit targets and visible references, then transform only system guidance."""
    by_action = defaultdict(deque)
    for event in events:
        by_action[(event['op'], event['assistant_content_sha256'])].append(event)
    transformed, provenance = [], []
    for record in requests:
        if record.get('native_token_ids') is not True:
            raise ValueError('native_token_ids_required')
        if record.get('requested_model') != model or record.get('returned_model') != model:
            raise ValueError('teacher_model_mismatch')
        if record.get('sft_assistance'):
            raise ValueError('scheduled_assistance_not_autonomous')
        original_ids = token_list(record.get('input_ids'))
        output = token_list(record.get('output_ids'))
        if execution_text(tokenizer, output, explicit=True) != record['response']:
            raise ValueError('native_output_text_mismatch')
        payload = json.loads(record['response'])  # Schema already validated by audit_episode.
        messages, removed = strip_guidance(record['messages'])
        if payload['tool'] in GRAPH_TOOLS:
            key = (payload['tool'], hashlib.sha256(record['response'].encode()).hexdigest())
            if not by_action[key]:
                raise ValueError('graph_action_request_mismatch')
            event = by_action[key].popleft()
            nodes = {n['id']: n for n in event['before_state']['nodes']}
            visible = '\n'.join(m['content'] for m in messages)
            for node in referenced_nodes(payload):
                if node not in nodes or not re.search(r'(?<![\w])' + re.escape(node) + r'(?![\w])', visible):
                    raise ValueError('graph_reference_not_in_visible_context')
            # Bind trace arguments to the actual sampled decision, not just its op.
            for key, value in payload['arguments'].items():
                expected = ','.join(value) if key == 'node_ids' else str(value)
                actual = event['args'].get(key)
                if isinstance(actual, list):
                    actual = ','.join(actual)
                if str(actual) != expected:
                    raise ValueError('graph_trace_argument_mismatch')
        prefix = tokenizer.apply_chat_template(messages, tokenize=True,
                                              add_generation_prompt=True, enable_thinking=True, return_dict=False)
        item = dict(record, messages=messages, input_ids=list(prefix))
        transformed.append(item)
        provenance.append({'source_input_sha256': checksum(original_ids),
                           'source_output_sha256': checksum(output),
                           'student_input_sha256': checksum(prefix),
                           'removed_guidance_blocks': removed,
                           'input_changed': original_ids != list(prefix),
                           'action': payload['tool']})
    if any(by_action.values()):
        raise ValueError('unmatched_graph_events')
    return transformed, provenance


def validate_manifest(manifest, data, heldout, signature):
    settings = manifest['settings']
    if manifest.get('profile') != DISTILL_PROFILE or manifest.get('collection_guidance') != DISTILL_GUIDANCE:
        raise ValueError('Expected registered explicit distillation collection; legacy messages need a separate audited adapter')
    if settings.get('graph_protocol') != 'explicit_tools' or settings.get('transport') != 'vllm_tokens':
        raise ValueError('wrong_collection_protocol')
    if manifest.get('tokenizer_sha256') != signature:
        raise ValueError('collection_student_tokenizer_mismatch')
    if manifest.get('source_sha256') != file_hash(data) or manifest.get('heldout_sha256') != file_hash(heldout):
        raise ValueError('source_or_heldout_file_changed')
    plugin = manifest['config']['actor_rollout_ref']['rollout']['plugin']
    if plugin.get('graph_action_protocol') != 'explicit_tools' or plugin.get('structured_graph_controller'):
        raise ValueError('scheduled_graph_controller_not_allowed')
    if any(plugin.get(k, 0) for k in ('consolidation_interval', 'initial_consolidation_turn', 'auto_prune_max_active')):
        raise ValueError('automatic_graph_maintenance_not_allowed')
    if plugin.get('apply_chat_template_kwargs', {}).get('enable_thinking') is not True:
        raise ValueError('thinking_collection_required')
    if plugin.get('explicit_action_thinking') is not True:
        raise ValueError('explicit_thinking_interface_required')


def build(args, tokenizer):
    import pandas as pd
    if args.max_length < 1 or not 0 <= args.validation_fraction < 1 or not 0 <= args.control_ratio <= 1:
        raise ValueError('Invalid length, split or control ratio')
    signature = tokenizer_signature(tokenizer)
    data = pd.read_parquet(args.data).to_dict('records')
    heldout = pd.read_parquet(args.heldout).to_dict('records')
    # Recheck the actual final-evaluation source, including rejected/downsampled tasks.
    select_training_rows(data, heldout, 0, -1, args.seed)
    roots = [Path(r).resolve() for r in args.runs]
    if len(set(roots)) != len(roots):
        raise ValueError('duplicate_collection_root')
    manifests = [read_json(r / 'manifest.json') for r in roots]
    for manifest in manifests:
        validate_manifest(manifest, args.data, args.heldout, signature)
        if (not manifest['indices'] or any(type(i) is not int or not 0 <= i < len(data) for i in manifest['indices'])
                or len(set(manifest['indices'])) != len(manifest['indices'])
                or type(manifest['settings']['candidates']) is not int or manifest['settings']['candidates'] < 1):
            raise ValueError('invalid_collection_indices_or_candidates')
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use an empty export directory')
    output.mkdir(parents=True, exist_ok=True)
    buckets, counts, decisions, pairs = defaultdict(list), Counter(), [], set()
    for root, manifest in zip(roots, manifests):
        expected = {(i, c) for i in manifest['indices'] for c in range(manifest['settings']['candidates'])}
        seen = set()
        counts['planned_episodes'] += len(expected)
        for path in sorted(root.glob('episodes/*/episode.json')):
            ep = read_json(path)
            key = (ep['source_index'], ep['candidate'])
            if key not in expected or key in seen:
                raise ValueError('unexpected_or_duplicate_episode')
            seen.add(key)
            expected_hash = checksum(question(data[ep['source_index']]))
            if ep['question_hash'] != expected_hash or question(data[ep['source_index']]).startswith('scienceworld:') != (ep['benchmark'] == 'scienceworld'):
                raise ValueError('episode_question_identity_mismatch')
            decision = dict(run=str(root), episode=str(path.relative_to(root)), question_hash=expected_hash)
            counts['episodes_seen'] += 1
            try:
                trajectory = read_json(root / ep['trajectory'])
                requests = read_jsonl(root / ep['requests'])
                reason, audit = audit_episode(ep, trajectory, requests)
                if reason:
                    raise ValueError(reason)
                transformed, source_records = prepare_requests(requests, audit['events'], tokenizer, manifest['settings']['teacher_model'])
                rows = segments_from_requests(transformed, args.max_length, signature)
                split = split_for(expected_hash, args.seed, args.validation_fraction)
                source_episode_id = checksum([checksum(manifest), key, checksum(requests)])
                kept, pending_pairs = [], set()
                for row in rows:
                    context = row['agent_context']
                    identities = [(s['prefix_sha256'], s['output_sha256']) for s in context['spans']]
                    # Reject overlapping segments as a unit rather than alter their masks.
                    if any((expected_hash, *identity) in pairs | pending_pairs for identity in identities):
                        counts['duplicate_segments'] += 1
                        continue
                    for identity in identities:
                        pending_pairs.add((expected_hash, *identity))
                    row.update(sft_profile=PROFILE, task_id=expected_hash, question_hash=expected_hash,
                               benchmark=ep['benchmark'], source_index=ep['source_index'], candidate=ep['candidate'],
                               teacher_model=manifest['settings']['teacher_model'], kind=audit['kind'], split=split,
                               graph_action_protocol='explicit_tools', timing_source='model', enable_thinking=True,
                               tools=[], task_reward=1.0, causal_graph_benefit_verified=False,
                               representation='transformed_prefix_native_targets', exact_replay=False,
                               source_manifest_sha256=checksum(manifest),
                               source_episode_id=source_episode_id,
                               request_provenance=[source_records[s['request_index']] for s in context['spans']],
                               trajectory_id=checksum([checksum(manifest), key, context['sha256']]))
                    validate_agent_context(row, args.max_length, signature)
                    kept.append(row)
                pairs.update(pending_pairs)
                buckets[f"{audit['kind']}_{split}"].extend(kept)
                actions = Counter(p['action'] for p in source_records)
                counts[f'accepted_source_episodes:{split}'] += 1
                decision.update(accepted=True, split=split, kind=audit['kind'], rows=len(kept), actions=dict(actions))
            except (ValueError, KeyError, TypeError) as exc:
                counts['rejected:' + str(exc)] += 1
                decision.update(accepted=False, reason=str(exc))
            decisions.append(decision)
        counts['incomplete_episodes'] += len(expected - seen)
    buckets['train'] = mix_training(buckets['graph_train'], buckets['control_train'], args.control_ratio, args.seed)
    buckets['validation'] = buckets['graph_validation'] + buckets['control_validation']
    coverage = {}
    for split in ('train', 'validation'):
        selected = buckets[split]
        coverage[split] = dict(
            rows=len(selected), questions=len({r['question_hash'] for r in selected}),
            episodes=len({r['source_episode_id'] for r in selected}),
            prune_episodes=len({r['source_episode_id'] for r in selected if any(p['action'] == 'prune' for p in r['request_provenance'])}),
            operation_calls=dict(Counter(p['action'] for r in selected for p in r['request_provenance'])))
    files = {}
    for name, rows in buckets.items():
        if rows:
            path = output / (name + '.parquet')
            pd.DataFrame(rows).to_parquet(path, index=False)
            files[path.name] = file_hash(path)
    report = dict(profile=PROFILE, counts=dict(counts), outputs={k: len(v) for k, v in buckets.items() if v},
                  files_sha256=files, tokenizer_sha256=signature, max_length=args.max_length,
                  teacher_checkpoints=[m.get('local_teacher_checkpoint') for m in manifests],
                  source_manifest_sha256=[checksum(m) for m in manifests],
                  heldout_sha256=file_hash(args.heldout), data_sha256=file_hash(args.data),
                  evaluation_overlap=0, evaluation_overlap_scope='normalized task text or ScienceWorld task/variation against supplied heldout file',
                  loaded_coverage=coverage,
                  training_ready=bool(buckets['train']), validation_ready=bool(buckets['validation']),
                  settings=vars(args), exact_replay=False, evidence_review_performed=False,
                  scope='successful explicit demonstrations; guidance removed; trace legality checked; no causal edit claim')
    (output / 'decisions.jsonl').write_text(''.join(json.dumps(d) + '\n' for d in decisions), encoding='utf-8')
    (output / 'manifest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    if args.require_complete and counts['incomplete_episodes']:
        raise ValueError('collection_incomplete')
    if not buckets['train'] and not args.smoke:
        raise ValueError('No graph training rows; inspect decisions without changing the split')
    if args.smoke and not any(d['accepted'] for d in decisions):
        raise ValueError('No accepted explicit distillation episodes')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+')
    parser.add_argument('--data', required=True)
    parser.add_argument('--heldout', required=True, help='Actual independent evaluation parquet; overlap fails before export')
    parser.add_argument('--tokenizer', required=True, help='Student tokenizer; fingerprint must match collection tokenizer')
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-length', type=int, default=65536)
    parser.add_argument('--validation-fraction', type=float, default=0.1)
    parser.add_argument('--control-ratio', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    if args.max_length < 1 or not 0 <= args.validation_fraction < 1:
        parser.error('Invalid length or split')
    from transformers import AutoTokenizer
    build(args, AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True))


if __name__ == '__main__':
    main()
