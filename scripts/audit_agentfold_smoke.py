"""Inspect saved AgentFold mechanics; do not gate on benchmark accuracy."""
import argparse
import collections
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluation_records import read_evaluation


def audit(root):
    root = Path(root)
    manifests, results, summary = read_evaluation(root)
    if manifests[0]['method'] != 'agentfold':
        raise ValueError('Expected an AgentFold evaluation')
    tasks = []
    for row in results:
        idx = row['source_index']
        d = json.loads((root / f'trajectory-{idx}.json').read_text(encoding='utf-8'))
        stats = d['env_stats']
        records = d.get('model_contexts', [])
        bad = [r for r in records if r.get('format_error')]
        tasks.append(dict(source_index=idx, stop=d.get('termination_reason'),
                          environment_steps=stats.get('environment_steps', 0),
                          folds=stats.get('agentfold_folds', 0),
                          format_errors=dict(collections.Counter(r['format_error'] for r in bad)),
                          format_errors_at_output_limit=stats.get('format_errors_at_output_limit'),
                          last_bad_response=bad[-1].get('response', '')[-1600:] if bad else None))
    checks = dict(no_execution_errors=summary['execution_errors'] == 0,
                  no_judge_errors=summary['judge_parse_failures'] == 0,
                  no_format_stops=all(t['stop'] != 'invalid_tool_limit' for t in tasks),
                  tools_executed_on_every_task=all(t['environment_steps'] > 0 for t in tasks),
                  finish_exercised=any(t['stop'] == 'finish' for t in tasks),
                  folding_exercised=any(t['folds'] > 0 for t in tasks))
    report = dict(scope='integration smoke only, not a performance claim',
                  passed=all(checks.values()), checks=checks, summary=summary, tasks=tasks)
    (root / 'agentfold-smoke-audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_root')
    report = audit(parser.parse_args().run_root)
    print(json.dumps(report, indent=2))
    sys.exit(0 if report['passed'] else 2)
