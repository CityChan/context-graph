"""Audit native task grading and memory mechanics; correctness is not a smoke gate."""
import argparse
import json
from pathlib import Path


def audit(root, benchmark, method, expected):
    root = Path(root)
    parts = sorted(root.glob('pair-*' if benchmark == 'swe-lite' else method + '-*'))
    tasks = []
    summaries = []
    for part in parts:
        if (part / 'summary.json').is_file():
            summaries.append(json.loads((part / 'summary.json').read_text()))
        for path in sorted(part.glob('instances/*/result.json')):
            result = json.loads(path.read_text())
            attempt = path.parent / Path(result.get('attempt', '')).name
            files = list(attempt.glob('generation/instances/*/trajectory.json')) if benchmark == 'swe-lite' else list(attempt.glob('trajectory.json'))
            trajectory = json.loads(files[0].read_text()) if len(files) == 1 else {}
            if isinstance(trajectory, list):
                trajectory = trajectory[0] if len(trajectory) == 1 else {}
            stats = trajectory.get('env_stats', {})
            tasks.append({'task': path.parent.name, 'status': result['status'],
                          'stop': trajectory.get('termination_reason'), 'stats': stats,
                          'provenance': trajectory.get(method),
                          'trajectory_present': bool(trajectory)})
    selected = sum(s.get('selected', 0) for s in summaries)
    checks = {
        'complete': len(parts) == len(summaries) == 2 and selected > 0
                    and len(tasks) == selected and (expected == -1 or selected == expected)
                    and all(s.get('pending', 0) == 0 for s in summaries),
        'native_grading_completed': bool(tasks) and all(t['status'] == 'graded' for t in tasks),
        'trajectory_and_provenance': bool(tasks) and all(t['trajectory_present'] and t['provenance'] for t in tasks),
        'no_format_stops': all(t['stop'] not in ('invalid_summary', 'invalid_tool_limit') for t in tasks),
        'tools_exercised': bool(tasks) and all(t['stats'].get('python_exec' if benchmark == 'swe-lite' else 'environment_steps', 0) > 0 for t in tasks),
        'memory_exercised': any(t['stats'].get('agentfold_folds' if method == 'agentfold' else 'summary_restarts', 0) > 0 for t in tasks),
    }
    report = {'scope': 'integration mechanics only; no accuracy claim', 'benchmark': benchmark,
              'method': method, 'passed': all(checks.values()), 'checks': checks, 'tasks': tasks}
    (root / 'stateful-memory-smoke-audit.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('benchmark', choices=['swe-lite', 'discoveryworld'])
    p.add_argument('method', choices=['agentfold', 'supo'])
    p.add_argument('--expected', type=int, default=2)
    args = p.parse_args()
    report = audit(args.root, args.benchmark, args.method, args.expected)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['passed'] else 2)
