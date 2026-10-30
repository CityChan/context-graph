"""Bounded structural proxies for model-owned joint graph management.

These scores describe graph structure/resource use, not evidence correctness.
Counters live on the episode's shared graph, so branch creation cannot reset caps.
"""
from collections import Counter
from dataclasses import dataclass, field

from .context_graph import NodeType


@dataclass
class ManagementLedger:
    credit: float = 0.0
    compact_bonuses: int = 0
    successful_operations: Counter = field(default_factory=Counter)
    effective_operations: Counter = field(default_factory=Counter)
    credited_operations: Counter = field(default_factory=Counter)
    model_edges: set = field(default_factory=set)
    model_edge_additions: int = 0


def ledger(graph):
    if not hasattr(graph, 'management_reward_ledger'):
        graph.management_reward_ledger = ManagementLedger()
    return graph.management_reward_ledger


def edge_key(edge):
    return edge['source'], edge['target'], edge['relation']


def record_operation(graph, agent, event, context_before, context_after):
    state = ledger(graph)
    credit = 0.0
    op = event['op']
    if event['source'] != 'model' or not event['success']:
        return 0.0
    state.successful_operations[op] += 1
    before, after = event['before_state'], event['after_state']
    before_active = sum(n['status'] == 'active' for n in before['nodes'])
    after_active = sum(n['status'] == 'active' for n in after['nodes'])
    if op in {'merge', 'prune'} and after_active < before_active:
        credit = .1
    elif op == 'add_edge':
        added = {edge_key(e) for e in after['edges']} - {edge_key(e) for e in before['edges']}
        if added:
            state.model_edges.update(added)
            state.model_edge_additions += len(added)
            credit = .05
    elif op == 'graph_compact':
        window = getattr(agent, 'working_context_limit', None) or (agent.config.prompt_length + agent.config.response_length)
        if context_before <= .75 * window and context_before - context_after >= 2000:
            credit = .2
    if credit:
        state.effective_operations[op] += 1
        if op == 'graph_compact' and state.compact_bonuses >= 2:
            credit = 0.0
        credit = min(credit, max(0.0, 1.0 - state.credit))
        if credit > 1e-12:
            state.credit = min(1.0, state.credit + credit)
            state.credited_operations[op] += 1
            state.compact_bonuses += int(op == 'graph_compact')
        else:
            credit = 0.0
    event['management_credit'] = credit
    event['management_credit_total'] = state.credit
    return credit


def terminal_reward(graph, task_reward, *, full=False, natural_finish=True, total_tokens=0):
    if task_reward not in (0., 1.):
        raise ValueError('Graph management requires a binary native task outcome')
    state = ledger(graph)
    observations = [n for n in graph.nodes.values() if n.type == NodeType.OBSERVATION]
    active = len(graph.active_nodes)
    archive_ratio = sum(not n.is_active() for n in observations) / max(len(observations), 1)
    size_score = min(1., 12. / max(active, 1))
    active_edges = {(e.source, e.target, e.relation.value) for e in graph.active_edges}
    model_active_edges = len(active_edges & state.model_edges)
    density_count = state.model_edge_additions if full else model_active_edges
    density_score = min(1., density_count / max(active - 1, 1))
    graph_score = .4 * archive_ratio + .4 * size_score + .2 * density_score if len(observations) >= 4 else 0.
    bonus = .2 * graph_score
    discounted_task = float(task_reward) * (.5 if full and not natural_finish else 1.)
    efficiency = .3 * discounted_task * max(0., 1. - max(0, total_tokens) / 3_000_000) if full else 0.
    metrics = dict(graph_management_score=graph_score, graph_management_terminal_bonus=bonus,
        graph_management_archive_ratio=archive_ratio, graph_management_size_score=size_score,
        graph_management_density_score=density_score, graph_management_observations=len(observations),
        graph_management_active_nodes=active, graph_management_model_active_edges=model_active_edges,
        graph_management_local_credit=state.credit, graph_management_compact_bonuses=state.compact_bonuses)
    metrics.update(graph_management_model_edge_additions=state.model_edge_additions,
        graph_management_discounted_task=discounted_task, graph_management_efficiency_bonus=efficiency)
    for op in ('merge', 'prune', 'add_edge', 'graph_compact'):
        metrics['graph_management_successful_' + op] = state.successful_operations[op]
        metrics['graph_management_effective_' + op] = state.effective_operations[op]
        metrics['graph_management_credited_' + op] = state.credited_operations[op]
    return discounted_task + bonus + efficiency, metrics
