import asyncio
import copy
import json
import re
from itertools import product

import numpy as np
import pytest

from agents import memory_baseline_agent as runner
from agents.memory_baseline_control import editable_node_ids, native_recall_constraint, native_recall_decision
from tests.test_graph_memory_baselines import graph, PAIR, PATCH, setup


def fold(group):
    return dict(fold=group, rationale='completed', notes=copy.deepcopy(PAIR))


def test_all_small_assignments_allow_arbitrary_disjoint_partitions():
    state = graph(7).state()
    ids = editable_node_ids(state)
    assert ids == [3, 4, 5, 6]
    regex = native_recall_constraint(state)['regex']
    # Keep, flush, or join any of four fold groups: covers every partition.
    for choices in product(range(6), repeat=4):
        nodes = {str(n): None if choice == 0 else dict(flush='old') if choice == 1
                 else fold(choice - 1) for n, choice in zip(ids, choices)}
        decision = dict(nodes=nodes)
        assert re.fullmatch(regex, json.dumps(decision))
        operations = native_recall_decision(decision, state)
        targets = [op['id'] for op in operations['flush_ops']]
        targets += [n for op in operations['fold_ops'] for n in op['ids']]
        assert len(targets) == len(set(targets))
        assert set(targets) == {n for n, choice in zip(ids, choices) if choice}
        for group in range(1, 5):
            expected = [n for n, choice in zip(ids, choices) if choice == group + 1]
            actual = [op for op in operations['fold_ops'] if op['ids'] == expected]
            assert len(actual) == bool(expected)
            if expected:
                assert actual[0]['notes'] == PAIR * len(expected)
        graph(7).maintain(operations)


def test_duplicate_keys_mixed_actions_and_bad_targets_are_not_decodable():
    state = graph().state()
    regex = native_recall_constraint(state)['regex']
    good = json.dumps(dict(nodes={'3': dict(flush='old'), '4': fold(1), '5': None}))
    assert re.fullmatch(regex, good)
    for bad in [good.replace('"3":', '"1":'), good.replace('"3":', '"4":'),
                good.replace('"5": null', '"5": null, "3": {"flush":"duplicate"}'),
                good.replace('"flush": "old"', '"flush": "old", "fold": 1'),
                good.replace('"fold": 1', '"fold": 0'),
                json.dumps(dict(flush_ops=[dict(id=3), dict(id=3)], fold_ops=[dict(ids=[3, 4], notes=PAIR)]))]:
        assert not re.fullmatch(regex, bad)


def test_noop_and_fresh_protection_and_linear_size():
    empty = graph(2).state()
    assert re.fullmatch(native_recall_constraint(empty)['regex'], '{"nodes":{}}')
    assert native_recall_decision(dict(nodes={}), empty) == dict(flush_ops=[], fold_ops=[])
    store = graph()
    before = store.state()
    decision = dict(nodes={'3': fold(1), '4': None, '5': dict(flush='old')})
    store.maintain(native_recall_decision(decision, before))
    assert not re.fullmatch(native_recall_constraint(store.state())['regex'], json.dumps(decision))
    for count in (10, 100):
        state = dict(nodes={i: dict(active=True) for i in range(count)}, protected=[])
        assert len(native_recall_constraint(state)['regex']) < count * 1200


def test_atomic_graph_cycle_validation_is_preserved():
    store = graph()
    store.edges += [dict(src=3, dst=4, rationale='dependency'), dict(src=4, dst=5, rationale='dependency')]
    before = copy.deepcopy(store.state())
    decision = dict(nodes={'3': fold(1), '4': None, '5': fold(1)})
    with pytest.raises(ValueError, match='cycle'):
        store.maintain(native_recall_decision(decision, before))
    assert store.state() == before


@pytest.mark.parametrize('invalid', [False, True])
def test_swe_recall_wire_conversion_is_audited_and_never_repairs_invalid_output(monkeypatch, invalid):
    action = '<function=python_exec><parameter=code>print(1)</parameter></function>'
    finish = '<function=finish><parameter=message>done</parameter></function>'
    decision = dict(nodes={'3': fold(1), '4': dict(flush='superseded')})
    text = json.dumps(decision)
    if invalid:
        text = text.replace('"3":', '"4":')  # Duplicate key, even if JSON parsing would collapse it.
    responses = [value for _ in range(5) for value in (action, json.dumps(PATCH))] + [text, finish]
    env, context, item, _ = setup(monkeypatch, 'memobrain', responses, observation='current file state')
    item.non_tensor_batch['ability'] = np.array(['SWEVerified@native'])
    context.config.actor_rollout_ref.rollout.plugin.workflow = 'code_memobrain'
    original = env.run_action
    async def run_action(call):
        result = await original(call)
        env.is_finish = call == finish
        return result
    env.run_action = run_action
    out = asyncio.run(runner.process_item(item, context))
    extra, stats = out.extra_fields, out.extra_fields['env_stats']
    assert extra['termination_reason'] == 'finish' and env.actions == [action] * 5 + [finish]
    assert stats['invalid_memory'] == int(invalid)
    assert stats['memory_requests'] == 6 and stats['model_requests'] == 12
    assert stats['memobrain_folds'] == stats['memobrain_flushes'] == int(not invalid)
    assert stats['main_len'] == sum(stats[k] for k in ('generated_tokens', 'observation_tokens', 'instruction_tokens'))
    record = extra['memory_audit'][-1]
    if not invalid:
        assert record['node_decision'] == decision and record['decision']['fold_ops'][0]['ids'] == [3]
    else:
        assert not record['applied'] and all(node['active'] for node in extra['memory_state']['nodes'].values())
    assert extra['memobrain']['recall_target_profile'] == 'node_assignment_regex_v1'
