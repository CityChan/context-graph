#!/usr/bin/env python3
"""Verify real server schema support before spending a paired evaluation run."""
import argparse
import json
from pathlib import Path
import sys

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


def main():
    from openai import OpenAI
    parser = argparse.ArgumentParser()
    parser.add_argument('--endpoint', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    checks = []
    with OpenAI(base_url=args.endpoint.rstrip('/') + '/v1', api_key='local', timeout=90, max_retries=0) as client:
        for name, schema in schemas():
            response = client.completions.create(model=args.model,
                prompt='Return one minimal JSON action satisfying the supplied schema. No explanation.\n',
                max_tokens=256, temperature=0,
                extra_body={'structured_outputs': {'json': schema}})
            value = json.loads(response.choices[0].text)
            _validate(value, schema)
            checks.append({'profile': name, 'passed': True})
    Path(args.output).write_text(json.dumps({'scope': 'schema mechanics only', 'checks': checks}, indent=2))
    print('GRAPH_RETRY_SCHEMA_PREFLIGHT_OK')


if __name__ == '__main__':
    main()
