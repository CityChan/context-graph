#!/usr/bin/env python3
"""Read existing rollout artifacts only; no model, tokenizer or network needed."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re


def diagnose(root, index, variant='baseline'):
    arm = Path(root) / variant
    trajectory = json.loads((arm / f'trajectory-{index}.json').read_text(encoding='utf-8'))
    contexts = trajectory.get('model_contexts', [])
    requests = []
    path = arm / f'requests-{index}.jsonl'
    if path.exists():
        with path.open(encoding='utf-8') as handle:
            for line in handle:
                row = json.loads(line)
                requests.append({'finish_reason': row.get('finish_reason'),
                                 'output_tokens': len(row.get('output_ids', [])),
                                 'max_tokens': row.get('max_tokens')})
    samples = []
    for turn in sorted(set([0, 1, len(contexts) - 1])):
        if not 0 <= turn < len(contexts):
            continue
        record = contexts[turn]
        text = str(record.get('response', ''))
        users = [m.get('content', '') for m in record.get('messages', []) if m.get('role') == 'user']
        samples.append({'request_in_main': turn + 1,
                        'output_tokens': len(record.get('output_ids', [])),
                        'response_head': text[:240], 'response_tail': text[-400:],
                        'preceding_user_tail': str(users[-1])[-400:] if users else '',
                        'has_think_close': '</think>' in text,
                        'complete_xml_call_names': re.findall(r'<function=([^>]+)>.*?</function>', text, re.S),
                        'action_protocol_error': record.get('action_protocol_error')})
    stats = trajectory.get('env_stats', {})
    return {'variant': variant, 'source_index': index,
            'stats': {k: stats.get(k) for k in ('search', 'open_page', 'session_time', 'hit_timeout', 'finalizer_attempted', 'forced_finish')},
            'requests_including_branches': len(requests), 'main_context_records': len(contexts),
            'finish_reasons': dict(Counter(str(r['finish_reason']) for r in requests)),
            'at_output_limit': sum(r['max_tokens'] is not None and r['output_tokens'] >= r['max_tokens'] for r in requests),
            'samples': samples,
            'scope': 'saved evidence; XML matches are syntactic indicators, not proof of execution'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run')
    parser.add_argument('--indices', nargs='+', type=int, default=[28, 70])
    parser.add_argument('--variant', choices=['baseline', 'guarded'], default='baseline')
    args = parser.parse_args()
    for index in args.indices:
        print(json.dumps(diagnose(args.run, index, args.variant), ensure_ascii=False, indent=2))
