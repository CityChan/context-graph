#!/usr/bin/env python3
"""Verify real server schema support before spending a paired evaluation run."""
import argparse
import json
import os
import re
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.action_contract import ActionContract, _validate
from agents.context_graph import ContextGraph, NodeType
from agents.graph_controller import GraphActionController
from agents.graph_retry_guard import GraphRetryGuard


def schemas():
    graph = ContextGraph()
    graph.add_node('Determine the published date.', NodeType.QUERY)
    graph.add_node('Source A reports a date.', NodeType.OBSERVATION)
    graph.add_node('Source B confirms the date.', NodeType.OBSERVATION)
    guard = GraphRetryGuard(graph)
    controller = GraphActionController(retry_guard=guard)
    contract = ActionContract('LocalSearch', None, retry_guard=guard)
    yield 'controller', controller.structured_outputs(controller.snapshot(graph), allow_pass=True)['json']
    yield 'explicit', contract.schema()
    guard.consecutive_failures = guard.failure_limit
    yield 'cooldown', contract.schema()


def task_probe(tokenizer, role, final):
    from agents.task_action_recovery import constraint
    if final:
        example = '<function=finish><parameter=answer>Unknown</parameter></function>'
    elif role == 'branch':
        example = '<function=return><parameter=message>No evidence yet.</parameter></function>'
    else:
        example = '<function=search><parameter=query>film critic</parameter></function>'
    messages = [
        {'role': 'system', 'content': 'This is a tool-format preflight. Copy the requested XML exactly, with no explanation.'},
        {'role': 'user', 'content': example},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=True,
                                          add_generation_prompt=True, enable_thinking=False)
    return {'prompt': prompt, 'max_tokens': 512, 'extra_body': {
        'structured_outputs': constraint(role, final),
        'return_token_ids': True, 'skip_special_tokens': False}}


def run_checks(client, model, output, tokenizer=None):
    """Persist every probe, including failures; never retry or relax validation."""
    checks = []
    probes = [(name, dict(prompt='Return one minimal JSON action satisfying the supplied schema. No explanation.\n',
                         max_tokens=256, extra_body={'structured_outputs': {'json': schema}}))
              for name, schema in schemas()]
    if tokenizer is not None:
        probes += [(f'task_recovery_{role}_{final}', task_probe(tokenizer, role, final))
                   for role, final in [('main', False), ('branch', False), ('main', True)]]
    for name, request in probes:
        started = time.monotonic()
        record = {'profile': name, 'passed': False, 'request': request}
        try:
            response = client.completions.create(model=model, temperature=0, **request)
            choice = response.choices[0]
            usage = getattr(response, 'usage', None)
            ids = getattr(choice, 'token_ids', None)
            record.update(response=choice.text, finish_reason=choice.finish_reason,
                          output_ids=ids, usage=usage.model_dump() if hasattr(usage, 'model_dump') else usage)
            schema = request['extra_body']['structured_outputs']
            if 'regex' in schema:
                if not isinstance(ids, list) or not ids:
                    raise ValueError('Missing generated token IDs (required by the evaluator)')
                # Match the real TokenClient decoding, not server text with special tokens.
                text = tokenizer.decode(ids, skip_special_tokens=True)
                record.update(decoded_response=text,
                              raw_output_text=tokenizer.decode(ids, skip_special_tokens=False))
                if len(ids) > request['max_tokens'] or re.fullmatch(schema['regex'], text) is None:
                    raise ValueError('Task recovery output is incomplete or violates the native XML schema')
            else:
                _validate(json.loads(choice.text), schema['json'])
            record['passed'] = True
        except Exception as exc:
            record['error'] = f'{type(exc).__name__}: {exc}'
            print('GRAPH_RETRY_SCHEMA_PREFLIGHT_FAILED ' + json.dumps({
                'profile': name, 'error': record['error'],
                'finish_reason': record.get('finish_reason'),
                'output_tokens': len(record.get('output_ids') or []),
                'response_tail': record.get('decoded_response', record.get('response', ''))[-600:],
                'artifact': str(output)}, ensure_ascii=False))
            raise
        finally:
            record['elapsed_seconds'] = time.monotonic() - started
            checks.append(record)
            Path(output).write_text(json.dumps({'scope': 'schema mechanics only', 'model': model,
                                               'checks': checks}, indent=2, ensure_ascii=False), encoding='utf-8')
    return checks


def main():
    from openai import OpenAI
    parser = argparse.ArgumentParser()
    parser.add_argument('--endpoint', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--tokenizer', help='Local checkpoint used by the evaluator; required for task recovery')
    args = parser.parse_args()
    tokenizer = None
    if os.environ.get('EVAL_ACTION_RECOVERY') == '1':
        if not args.tokenizer:
            parser.error('--tokenizer is required with EVAL_ACTION_RECOVERY=1')
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    with OpenAI(base_url=args.endpoint.rstrip('/') + '/v1', api_key='local', timeout=90, max_retries=0) as client:
        run_checks(client, args.model, args.output, tokenizer)
    print('GRAPH_RETRY_SCHEMA_PREFLIGHT_OK')


if __name__ == '__main__':
    main()
