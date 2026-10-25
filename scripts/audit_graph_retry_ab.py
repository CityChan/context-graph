#!/usr/bin/env python3
"""Paired artifact audit. Completeness is distinct from accuracy or guard exercise."""
import argparse
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluation_records import read_evaluation, task_tool_count


def audit_task_recovery(root):
    root = Path(root)
    manifests, rows, summary = read_evaluation(root)
    plugin = manifests[0]['config']['actor_rollout_ref']['rollout']['plugin']
    assert plugin.get('inference_task_action_recovery'), 'recovery not enabled'
    tasks = [{'source_index': row['source_index'], 'status': row['status'],
              'task_tools': task_tool_count(row),
              'retry_requests': row.get('env_stats', {}).get('task_format_retry_requests', 0),
              'valid_retry_calls': row.get('env_stats', {}).get('task_format_recovery_successes', 0),
              'failed_retry': row.get('env_stats', {}).get('hit_format_retry_limit', 0)} for row in rows]
    complete = all((root / f"trajectory-{r['source_index']}.json").is_file() for r in rows)
    exercised = any(t['valid_retry_calls'] for t in tasks)
    passed = bool(tasks) and complete and exercised and all(t['status'] == 'ok' and (t['task_tools'] or 0) > 0 and not t['failed_retry'] for t in tasks)
    result = {'scope': 'task-action recovery mechanics only; not a graph-guard A/B or accuracy claim',
              'passed': passed, 'recovery_exercised': exercised, 'tasks': tasks}
    (root / 'task-recovery-audit.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def normalized_manifest(manifest):
    result = copy.deepcopy(manifest)
    plugin = result['config']['actor_rollout_ref']['rollout']['plugin']
    plugin.pop('graph_retry_guard', None)
    return result


def audit(root):
    root = Path(root)
    reports, manifests, identities = {}, {}, {}
    for variant, enabled in [('baseline', False), ('guarded', True)]:
        directory = root / variant
        arm_manifests = [json.loads((directory / f'manifest-{rank}.json').read_text()) for rank in range(3)]
        for rank, manifest in enumerate(arm_manifests):
            assert manifest['rank'] == rank, 'wrong manifest rank'
            rollout = manifest['config']['actor_rollout_ref']['rollout']
            assert rollout['plugin']['graph_retry_guard'] is enabled, 'wrong guard setting'
            assert rollout['prompt_length'] + rollout['response_length'] == 65536, 'not a 64K run'
            assert rollout['plugin']['val_response_length'] == 57344, 'validation overrides 64K'
        assert all(m['indices'] == arm_manifests[0]['indices'] for m in arm_manifests), 'shard selection mismatch'
        expected = arm_manifests[0]['indices']
        rows = []
        for rank in range(3):
            shard = [json.loads(line) for line in (directory / f'results-{rank}.jsonl').read_text().splitlines() if line.strip()]
            assert sorted(r['source_index'] for r in shard) == sorted(expected[rank::3]), 'incomplete/duplicate shard'
            rows.extend(shard)
        assert rows and all(r['status'] == 'ok' for r in rows), 'execution errors or no results'
        summary = json.loads((directory / 'summary.json').read_text())
        for key in ('execution_errors', 'judge_parse_failures', 'format_retry_failures'):
            assert summary.get(key, 0) == 0, key
        assert summary.get('generation_quality_passed') is True, 'generation quality not verified'
        for row in rows:
            assert (directory / f"trajectory-{row['source_index']}.json").is_file(), 'missing trajectory'
            assert row['env_stats'].get('token_budget_mode') == 'working_context'
            if enabled:
                assert row['env_stats'].get('graph_retry_guard', {}).get('enabled') is True, 'guard not installed'
        total = lambda key: sum(r['env_stats'].get(key, 0) for r in rows)
        guard_total = lambda key: sum(r['env_stats'].get('graph_retry_guard', {}).get(key, 0) for r in rows)
        reports[variant] = {
            'count': len(rows), 'successes': sum(r['task_reward'] for r in rows),
            'total_model_tokens': total('total_model_tokens'),
            'graph_attempts_reported': total('graph_op_attempts'),
            'graph_invalid_ops_reported': total('graph_invalid_ops'),
            'controller_errors': total('consol_controller_errors'),
            'guard_requests': guard_total('requests'), 'guard_failures': guard_total('failures'),
            'guard_input_tokens': guard_total('input_tokens'), 'guard_output_tokens': guard_total('output_tokens'),
            'failed_input_tokens': guard_total('failed_input_tokens'),
            'failed_output_tokens': guard_total('failed_output_tokens'),
            'blocked_attempts': guard_total('blocked_attempts'), 'cooldowns': guard_total('cooldowns'),
            'zero_tool_tasks': sum(task_tool_count(r) == 0 for r in rows),
            'unknown_tool_counts': sum(task_tool_count(r) is None for r in rows),
            'no_tool_limit_tasks': total('hit_no_tool_limit'),
        }
        identities[variant] = sorted((r['source_index'], r['task_id']) for r in rows)
        manifests[variant] = [normalized_manifest(m) for m in arm_manifests]
    assert identities['baseline'] == identities['guarded'], 'unpaired tasks'
    assert manifests['baseline'] == manifests['guarded'], 'code/model/seed/data/protocol differs beyond guard flag'
    result = {
        'scope': 'paired integration smoke; no benchmark improvement claim',
        'passed': all(not r['zero_tool_tasks'] and not r['unknown_tool_counts'] and not r['no_tool_limit_tasks'] for r in reports.values()),
        'guard_exercised': reports['guarded']['guard_requests'] > 0,
        'cooldown_exercised': reports['guarded']['cooldowns'] > 0,
        'failure_guard_exercised': reports['guarded']['guard_failures'] > 0,
        'accounting_note': 'Baseline controller failures are undercounted in graph_invalid_ops; compare controller_errors and saved traces as well.',
        'timing_note': 'Servers are reused; guarded runs second with warm caches. Do not infer latency improvement.',
        'arms': reports,
    }
    (root / 'retry-ab-audit.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root')
    parser.add_argument('--task-recovery-only', action='store_true')
    args = parser.parse_args()
    result = audit_task_recovery(args.root) if args.task_recovery_only else audit(args.root)
    print(json.dumps(result, indent=2))
    sys.exit(0 if result['passed'] else 1)
