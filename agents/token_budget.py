"""Evaluation budget semantics, independent of the lifetime token ledger.

Training and old configs retain their historical behavior. New evaluators opt
into a per-request working window; compression may reclaim that space, but never
rewrites the sampled trajectory or refunds recorded computation.
"""

import os
from dataclasses import dataclass


def window_budget(config):
    mode = getattr(config.plugin, 'token_budget_mode', 'legacy_cumulative')
    if mode not in ('working_context', 'legacy_cumulative'):
        raise ValueError(f'Unknown token_budget_mode: {mode}')
    return mode == 'working_context'


def configure_budget(config):
    """Resolve the launch override into the saved config, never inside a worker."""
    rollout = config.actor_rollout_ref.rollout
    rollout.plugin.token_budget_mode = os.environ.get('AGENT_BUDGET_MODE', 'working_context')
    window_budget(rollout)
    return config


@dataclass
class TokenBudget:
    window: int
    cumulative_limit: int
    use_window: bool

    @classmethod
    def from_config(cls, config):
        return cls(int(config.prompt_length) + int(config.response_length),
                   int(getattr(config.plugin, 'val_response_length', None) or config.response_length),
                   window_budget(config))

    def remaining(self, used, context_tokens=0):
        return max(0, self.window - context_tokens if self.use_window else self.cumulative_limit - used)

    def generation_limit(self, used, context_tokens, instruction_cost=0, reserve=0):
        # Instructions are already in the rendered input in window mode.
        if self.use_window:
            return max(0, self.window - context_tokens - reserve)
        return max(0, min(self.window - context_tokens,
                          self.cumulative_limit - used - instruction_cost - reserve))

    def metadata(self):
        return {'token_budget_mode': 'working_context' if self.use_window else 'legacy_cumulative',
                'working_context_limit': self.window,
                'episode_token_limit': None if self.use_window else self.cumulative_limit,
                'budget': ('per-request rendered input + output within working context; '
                           'lifetime consumption recorded, not capped by context size'
                           if self.use_window else 'legacy cumulative response budget; no refunds')}

    def stats(self, used, audit, reserve):
        final_calls = [r for r in audit if r.get('phase') == 'final']
        return {**self.metadata(), 'cumulative_accounted_tokens': used,
                'total_model_tokens': sum(len(r['input_ids']) + len(r['output_ids']) for r in audit),
                'peak_request_tokens': max((len(r['input_ids']) + len(r['output_ids']) for r in audit), default=0),
                'finalizer_enabled': int(reserve > 0),
                'finalizer_attempted': int(bool(final_calls)),
                'forced_finish': int(any('action' in r and not r.get('format_error') for r in final_calls))}


def agent_budget_stats(agents, config):
    """Record the protocol and lifetime cost across main and branch sessions."""
    agents = list(agents)
    return {**TokenBudget.from_config(config).metadata(),
            'response_ledger_tokens': sum(a.response_tokens_used() - getattr(a, 'external_response_tokens', 0) for a in agents),
            'total_model_tokens': sum(getattr(a, 'total_request_tokens', 0) for a in agents),
            'peak_request_tokens': max((getattr(a, 'peak_request_tokens', 0) for a in agents), default=0),
            'finalizer_enabled': int(bool(getattr(config.plugin, 'final_answer_reserve', 0)))}
