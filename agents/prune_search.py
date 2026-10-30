"""Opt-in binary outcome + successful prune credit + executor rejection labels."""
from collections import Counter


PROFILE = 'prune_search_v1'


def search_key(arguments, env):
    # Match LocalSearch's effective display top-k, including its default and cap.
    value = arguments.get('topk', 10)
    topk = int(value) if str(value).isdigit() else 10
    topk = min(max(topk, 1), env.search_topk_cap)
    query = ' '.join(str(arguments.get('query', '')).casefold().split())
    return query, topk


class PruneSearchRuntime:
    def __init__(self, graph, env, *, train):
        self.graph, self.env, self.train = graph, env, train
        self.searches = set()
        self.prunes = {}
        self.revoked = set()
        self.metrics = Counter()
        self.finished = False

    def register(self, agent):
        agent.prune_search_runtime = self

    def rejected(self, agent, reason):
        self.metrics[reason] += 1
        turn = len(agent.chat) - 1
        info = agent.additional_info[turn]
        if info is None:
            info = agent.additional_info[turn] = {}
        info['prune_search_rejection'] = reason
        if self.train:
            agent.set_process_reward(turn, -.3)

    def before_dispatch(self, agent, payload):
        if payload['tool'] == 'search' and search_key(payload['arguments'], self.env) in self.searches:
            self.rejected(agent, 'repeat_search')
            # The ordinary rejected-action path consumes a turn but creates no node.
            return ('[Action protocol rejected] Duplicate successful search blocked. '
                    'Use graph_read/recall for stored evidence or change the query/effective topk.')
        return None

    def observed(self, agent, payload):
        """Return whether this result supplies evidence; never infer success from prose."""
        outcome = getattr(self.env, 'last_action_outcome', {})
        if outcome.get('rejected'):
            self.rejected(agent, 'invalid_environment_action')
        if payload.get('tool') == 'search':
            if outcome.get('success'):
                self.searches.add(search_key(payload['arguments'], self.env))
                self.metrics['successful_searches'] += 1
            else:
                self.metrics['failed_searches'] += 1
            return bool(outcome.get('success') and outcome.get('new_documents', 0))
        return bool(outcome.get('success') and outcome.get('documents', 0))

    def edited(self, agent, event):
        if event['source'] != 'model':
            return
        if not event['success']:
            self.rejected(agent, 'invalid_graph_action')
            return
        node = event['args'].get('node_id')
        if event['op'] == 'recall' and node in self.prunes:
            self.revoked.add(node)
        if event['op'] == 'prune':
            before = sum(n['status'] == 'active' for n in event['before_state']['nodes'])
            after = sum(n['status'] == 'active' for n in event['after_state']['nodes'])
            if after < before and node not in self.prunes:
                self.prunes[node] = agent, event['assistant_turn_index']
                self.metrics['prune_candidates'] += 1
        if event['op'] == 'graph_compact':
            refresh = event.get('context_refresh', {})
            self.metrics['compact_tokens_saved'] += refresh.get('tokens_before', 0) - refresh.get('tokens_after', 0)

    def finish(self, reward):
        if reward not in (0., 1.):
            raise ValueError('prune_search_v1 requires binary native task reward')
        if self.finished:
            raise ValueError('prune credit must be settled only once')
        self.finished = True
        credited = 0
        if self.train and reward == 1:
            for node_id, (agent, turn) in self.prunes.items():
                if credited == 5:
                    break
                node = self.graph.nodes.get(node_id)
                if node_id not in self.revoked and node is not None and node.status.value == 'pruned':
                    agent.add_graph_edit_credit(turn, .1)
                    credited += 1
        self.metrics['credited_prunes'] = credited
        return self.stats()

    def stats(self):
        return {'prune_search_' + key: value for key, value in self.metrics.items()}
