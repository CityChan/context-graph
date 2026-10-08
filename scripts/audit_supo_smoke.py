"""Inspect SUPO summary/resume/finish mechanics, independently of answer accuracy."""
import argparse
import json
import re
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluation_records import read_evaluation


def repetitive_text(text):
    # Diagnostic only: a multiword block repeated four times without interruption.
    # This does not replace correctness grading or the token-zero audit.
    return bool(re.search(r'(.{20,300})\1{3,}', text, re.S))


def audit(root):
    root = Path(root)
    manifests, rows, summary = read_evaluation(root)
    if manifests[0]['method'] != 'supo':
        raise ValueError('Expected a SUPO evaluation')
    tasks = []
    for row in rows:
        idx = row['source_index']
        d = json.loads((root / f'trajectory-{idx}.json').read_text(encoding='utf-8'))
        stats = d['env_stats']
        records = d.get('model_contexts', [])
        tasks.append(dict(source_index=idx, stop=d.get('termination_reason'),
                          stats={k: stats.get(k) for k in ['environment_steps', 'summary_attempts',
                                 'summary_restarts', 'invalid_summary', 'hit_token_limit', 'hit_summary_limit']},
                          errors=[dict(phase=r.get('phase'), error=r['format_error'])
                                  for r in records if r.get('format_error')],
                          text_repetition_warnings=[dict(request=i, phase=r.get('phase'))
                              for i, r in enumerate(records) if repetitive_text(r.get('response', ''))],
                          last_response=records[-1].get('response', '')[-1200:] if records else None))
    checks = dict(no_execution_errors=summary['execution_errors'] == 0,
                  no_judge_errors=summary['judge_parse_failures'] == 0,
                  no_summary_format_failures=summary['summary_format_failures'] == 0,
                  no_tool_format_stops=summary['format_retry_failures'] == 0,
                  tools_exercised=all((t['stats']['environment_steps'] or 0) > 0 for t in tasks),
                  summary_resumed=any((t['stats']['summary_restarts'] or 0) > 0 for t in tasks),
                  finish_exercised=any(t['stop'] == 'finish' for t in tasks))
    report = dict(scope='integration smoke only, not benchmark performance', passed=all(checks.values()),
                  generation_quality_scope='shared generation_quality_passed checks token-zero runs only; text repetition warnings are separate',
                  checks=checks, summary=summary, tasks=tasks)
    (root / 'supo-smoke-audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_root')
    report = audit(parser.parse_args().run_root)
    print(json.dumps(report, indent=2))
    sys.exit(0 if report['passed'] else 2)
