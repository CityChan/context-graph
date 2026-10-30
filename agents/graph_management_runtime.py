"""Optional full graph-management recipe; all labels are auditable proxies."""
import re
from collections import Counter

from .explicit_graph_tools import compaction_plan, compact_context


FULL_PROFILE = 'graph_management_full_v1'


def normalized(value):
    return ' '.join(str(value).casefold().split())


def retrieval_key(tool, args):
    if tool == 'search':
        return tool, normalized(args.get('query', ''))
    if tool == 'open_page':
        return tool, normalized(args.get('docid', args.get('url', '')))
    return None


class GraphManagementRuntime:
    def __init__(self, graph, trace, plugin, *, train, answers=()):
        self.graph, self.trace, self.plugin, self.train = graph, trace, plugin, train
        self.roles, self.parents = {}, {}
        self.env_steps = 0
        self.last_edit_step = 0
        self.recent_edits = []
        self.discarded = {}
        self.retrievals = set()
        self.branch_prompts = []
        self.timeline = []
        self.answer_seen_at = None
        answer_values = [] if answers is None else ([answers] if isinstance(answers, str) else answers)
        self.answers = [normalized(a) for a in answer_values if str(a).strip()]
        self.auto_compacts = 0
        self.last_auto_step = -10**9
        self.metrics = Counter()

    def register(self, agent, name, *, parent=None, parent_turn=None, high_pressure=False):
        self.roles[id(agent)] = name
        if parent is not None:
            self.parents[id(agent)] = parent, parent_turn, high_pressure
        agent.graph_management_runtime = self

    def window(self, agent):
        return getattr(agent, 'working_context_limit', None) or agent.config.prompt_length + agent.config.response_length

    def label(self, agent, turn, value, reason):
        if not self.train or turn < 0 or agent.chat_completions[turn] is None:
            return
        info = agent.additional_info[turn]
        if info is None:
            info = agent.additional_info[turn] = {}
        info['process_reward'] = min(float(info.get('process_reward', 0.)), value)
        info.setdefault('graph_management_penalties', []).append({'reason': reason, 'value': value})
        self.metrics['penalty_' + reason] += 1

    def sampled(self, agent, payload, input_tokens):
        agent.gm_last_input_tokens = input_tokens
        tool, args = payload['tool'], payload['arguments']
        turn = len(agent.chat) - 1
        if tool == 'return' and agent.force_action_finish:
            parent = self.parents.get(id(agent))
            if parent is not None and not parent[2]:
                self.label(parent[0], parent[1], -.1, 'forced_branch_return')
                self.label(agent, turn, -.1, 'forced_branch_return')
        if tool == 'branch':
            words = set(normalized(args.get('description', '') + ' ' + args.get('prompt', '')).split())
            threshold = float(self.plugin.get('gm_branch_similarity_threshold', .8))
            if any(len(words & old) / max(len(words | old), 1) >= threshold for old in self.branch_prompts):
                self.label(agent, turn, -.3, 'duplicate_branch')
            self.branch_prompts.append(words)
            if len(self.branch_prompts) >= 5:
                self.label(agent, turn, -.1, 'excess_branch')
        if tool not in {'search', 'open_page', 'python_exec', 'action'}:
            return
        self.env_steps += 1
        self.timeline.append((self.env_steps, agent, turn))
        key = retrieval_key(tool, args)
        if key:
            if key in self.retrievals:
                self.label(agent, turn, -.2, 'repeat_' + tool)
            self.retrievals.add(key)
            for edit_agent, edit_turn in self.discarded.get(key, []):
                self.label(edit_agent, edit_turn, -.2, 'discarded_source_retrieved')
        if len(self.graph.active_nodes) > 12 and self.env_steps - self.last_edit_step > 10:
            self.label(agent, turn, -.1, 'graph_neglect')
        if input_tokens >= .9 * self.window(agent):
            self.label(agent, turn, -.05, 'context_pressure')
            for step, edit_agent, edit_turn in self.recent_edits:
                if 0 < self.env_steps - step <= 4:
                    self.label(edit_agent, edit_turn, -.2, 'pressure_rebound')

    def observation(self, agent, payload, text):
        if text is None:
            return
        tool = payload.get('tool')
        if tool in {'search', 'open_page', 'python_exec', 'action'}:
            agent.gm_pending_observation = True
            count = len(agent.tokenizer.encode(str(text), add_special_tokens=False))
            if self.roles[id(agent)] == 'main' and tool in {'search', 'open_page'} and count >= 1000:
                self.label(agent, len(agent.chat) - 1, -min(.2, count / self.window(agent)), 'large_main_observation')
            self.maintain(agent, pending_tokens=count)

    def visible_observation(self, agent, text):
        # Inspect the appended (possibly budget-truncated) feedback, not hidden
        # raw tool results. Gold answers are used for training labels only.
        if not getattr(agent, 'gm_pending_observation', False):
            return
        agent.gm_pending_observation = False
        observation = normalized(text)
        if self.answer_seen_at is None and any(re.search(r'(?<!\w)' + re.escape(a) + r'(?!\w)', observation) for a in self.answers):
            self.answer_seen_at = self.env_steps

    def edited(self, agent, event):
        if not event['success'] or event['source'] != 'model':
            return
        op = event['op']
        changed = event.get('semantic_before_hash') != event.get('semantic_after_hash')
        if op in {'merge', 'prune', 'add_edge', 'graph_compact'} and (changed or op == 'graph_compact'):
            self.last_edit_step = self.env_steps
            self.recent_edits.append((self.env_steps, agent, event['assistant_turn_index']))
            self.recent_edits = [e for e in self.recent_edits if self.env_steps - e[0] <= 4]
        if op in {'merge', 'prune'}:
            after = {n['id']: n for n in event['after_state']['nodes']}
            for node in event['before_state']['nodes']:
                if node['status'] == 'active' and after.get(node['id'], {}).get('status') != 'active':
                    meta = node.get('metadata', {})
                    key = retrieval_key(meta.get('tool'), meta.get('arguments', {}))
                    if key:
                        self.discarded.setdefault(key, []).append((agent, event['assistant_turn_index']))

    def finish(self, native_reward):
        if native_reward > 0 and self.answer_seen_at is not None:
            for step, agent, turn in self.timeline:
                if step - self.answer_seen_at > 8:
                    self.label(agent, turn, -.1, 'answer_delay')
        return {'graph_management_' + k: v for k, v in self.metrics.items()}

    def maintain(self, agent, pending_tokens=0):
        """Framework events never call execute_graph_tool or acquire policy credit."""
        if agent.force_action_finish:
            return
        limit = int(self.plugin.get('gm_auto_prune_max_active', 0))
        if limit:
            before = self.trace.capture(self.graph)
            pruned = self.graph.auto_prune_low_value(max_active=limit)
            if pruned:
                self.trace.record(self.graph, before, turn_id=self.env_steps, source='framework',
                    op='auto_prune', args={'node_ids': pruned}, decision_context={'agent': self.roles[id(agent)]})
                self.metrics['auto_pruned_nodes'] += len(pruned)
        if not self.plugin.get('gm_auto_compact', False):
            return
        before_tokens = len(agent.context())
        threshold = float(self.plugin.get('gm_auto_compact_threshold', .75))
        if before_tokens + pending_tokens < threshold * self.window(agent):
            return
        if (self.auto_compacts >= int(self.plugin.get('gm_auto_compact_max', 6))
                or self.env_steps - self.last_auto_step < int(self.plugin.get('gm_auto_compact_cooldown', 4))):
            return
        _, cost, _ = compaction_plan(agent, self.graph)
        predicted = cost + len(agent.get_generation_prompt())
        gain = (before_tokens - predicted) / max(before_tokens, 1)
        if gain < float(self.plugin.get('gm_auto_compact_min_gain', .25)):
            self.metrics['auto_compact_gain_skips'] += 1
            return
        before = self.trace.capture(self.graph)
        try:
            compact_context(agent, self.graph, source='framework')
        except ValueError:
            self.metrics['auto_compact_rejections'] += 1
            return
        self.auto_compacts += 1
        self.last_auto_step = self.env_steps
        event = self.trace.record(self.graph, before, turn_id=self.env_steps, source='framework',
            op='auto_compact', decision_context={'agent': self.roles[id(agent)]})
        event['context_refresh'] = {'tokens_before': before_tokens, 'tokens_after': len(agent.context())}
        self.metrics['auto_compacts'] += 1
