"""Opt-in inference protection. State lives outside the compressible transcript."""
from __future__ import annotations

import hashlib
import json
from collections import Counter

GRAPH_TOOLS = {'merge', 'prune', 'select', 'add_edge', 'recall', 'graph_read', 'graph_compact'}


def configured_guard(graph, plugin, *, is_train):
    enabled = str(getattr(plugin, 'graph_retry_guard', False)).lower()
    if enabled not in {'true', 'false', '1', '0'}:
        raise ValueError('graph_retry_guard must be boolean')
    if enabled in {'false', '0'}:
        return None
    if is_train:
        raise ValueError('graph_retry_guard is an inference experiment, not a training policy')
    guard = GraphRetryGuard(graph, int(getattr(plugin, 'graph_retry_failure_limit', 3)))
    graph.retry_guard = guard
    return guard


class GraphRetryGuard:
    def __init__(self, graph, failure_limit=3):
        if failure_limit < 1:
            raise ValueError('graph_retry_failure_limit must be positive')
        self.graph = graph
        self.failure_limit = failure_limit
        self.consecutive_failures = 0
        self.failures = {}
        self.counters = Counter()

    @property
    def task_only(self):
        return self.consecutive_failures >= self.failure_limit

    def signature(self, call):
        """Exclude diagnostic counters; unrelated observation nodes do not unlock a target."""
        args = dict(call.get('arguments', {}))
        ids = args.get('node_ids', [])
        if isinstance(ids, str):
            ids = [x.strip() for x in ids.split(',') if x.strip()]
        if ids:
            args['node_ids'] = sorted(ids)
        targets = set(ids)
        targets.update(args[k] for k in ('node_id', 'source', 'target') if k in args)
        nodes = {nid: (self.graph.nodes[nid].status.value, self.graph.nodes[nid].content)
                 if nid in self.graph.nodes else None for nid in targets}
        edges = [(e.source, e.target, e.relation.value) for e in self.graph.edges
                 if e.source in targets or e.target in targets]
        state = {'nodes': nodes, 'edges': sorted(edges)}
        if call.get('function') == 'select':
            state['focus'] = self.graph.active_node_id
        # Compaction has additional transcript-dependent preconditions; never cache
        # its rejection solely by graph state.
        key = json.dumps([call.get('function'), args], sort_keys=True, default=str)
        digest = hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()
        return key, digest

    def blocked(self, call):
        if self.task_only:
            return 'Graph editing paused after consecutive failures; execute a task tool first.'
        if call.get('function') != 'graph_compact':
            key, digest = self.signature(call)
            if self.failures.get(key) == digest:
                return 'This same operation already failed on unchanged targets; choose another action.'
        if call.get('function') == 'select' and call.get('arguments', {}).get('node_id') == self.graph.active_node_id:
            return 'That node is already the current focus; choose another action.'
        return None

    def record(self, call, success, *, input_tokens=0, output_tokens=0, blocked=False):
        self.counters['requests'] += 1
        self.counters['input_tokens'] += input_tokens
        self.counters['output_tokens'] += output_tokens
        if not success:
            self.counters['failures'] += 1
            self.counters['failed_input_tokens'] += input_tokens
            self.counters['failed_output_tokens'] += output_tokens
            self.counters['blocked_attempts'] += int(blocked)
            self.consecutive_failures += 1
            if call.get('function') != 'graph_compact':
                key, digest = self.signature(call)
                self.failures[key] = digest
            if self.consecutive_failures == self.failure_limit:
                self.counters['cooldowns'] += 1
        elif call.get('function') in {'merge', 'prune', 'add_edge', 'select', 'recall'}:
            self.consecutive_failures = 0

    def task_executed(self):
        # Only dispatch of a real task tool unlocks the cooldown, not a read,
        # compaction, branch handoff, parse retry or a successful controller pass.
        self.consecutive_failures = 0

    def feedback(self, error):
        candidates = [n.id for n in self.graph.active_nodes if n.id != self.graph.root_id][-12:]
        return (str(error).split('\n', 1)[0][:240] + '\nActive editable IDs: '
                + ', '.join(candidates) + ('. Execute a task tool next.' if self.task_only else '.'))

    def stats(self):
        return {'enabled': True, 'failure_limit': self.failure_limit,
                'consecutive_failures': self.consecutive_failures, **dict(self.counters)}


def request_cost(agent):
    records = getattr(agent, 'model_contexts', [])
    if not records:
        return {}
    record = records[-1]
    return {'input_tokens': len(record.get('input_ids', [])),
            'output_tokens': len(record.get('output_ids', []))}


def constrain_explicit_schema(schema, guard):
    """Simple dynamic enums supported by the existing typed-action decoder."""
    graph = guard.graph
    active = [n.id for n in graph.active_nodes]
    editable = [nid for nid in active if nid != graph.root_id]
    inactive = [n.id for n in graph.nodes.values() if not n.is_active()]
    choices = []
    for choice in schema['anyOf']:
        name = choice['properties']['tool']['const']
        if name in GRAPH_TOOLS and guard.task_only:
            continue
        props = choice['properties']['arguments']['properties']
        if name in {'merge', 'prune', 'select', 'recall'}:
            ids = (editable if name in {'merge', 'prune'} else inactive if name == 'recall'
                   else [nid for nid in active if nid != graph.active_node_id])
            if name != 'merge':
                ids = [nid for nid in ids if not guard.blocked({'function': name, 'arguments': {'node_id': nid}})]
            if len(ids) < (2 if name == 'merge' else 1):
                continue
            if name == 'merge':
                props['node_ids']['items'] = {'type': 'string', 'enum': ids}
                props['node_ids']['maxItems'] = min(6, len(ids))
            else:
                props['node_id'] = {'type': 'string', 'enum': ids}
        elif name == 'add_edge':
            # Each alternative binds a source/relation to legal distinct targets.
            import copy
            for source in active:
                for relation in ('causal', 'semantic', 'temporal'):
                    targets = [target for target in active if target != source and
                               not any(e.source == source and e.target == target and e.relation.value == relation for e in graph.edges)]
                    if not targets:
                        continue
                    edge = copy.deepcopy(choice)
                    ep = edge['properties']['arguments']['properties']
                    ep.update(source={'const': source}, relation={'const': relation},
                              target={'type': 'string', 'enum': targets})
                    choices.append(edge)
            continue
        elif name == 'graph_read':
            props['node_id'] = {'type': 'string', 'enum': list(graph.nodes)}
        choices.append(choice)
    return {'anyOf': choices}
