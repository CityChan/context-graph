"""Explicit, outcome-only joint RL with verified illegal-action penalties."""

PROFILE = 'invalid_action_v1'


def validate_profile(algorithm, plugin):
    profile = plugin.get('graph_rl_guard_profile', 'none')
    if profile == 'none':
        return
    if profile != PROFILE:
        raise ValueError('Unknown graph RL guard profile')
    if (str(algorithm.get('adv_estimator', '')).lower() not in {'graphrpo', 'advantageestimator.graphrpo'}
            or algorithm.get('graphrpo_invalid_action_policy') != 'negative_invalid_graph'
            or not algorithm.get('graphrpo_require_binary_reward', True)):
        raise ValueError('Guarded joint RL requires GraphRPO, binary rewards and negative_invalid_graph')
    if plugin.get('graph_action_protocol') != 'explicit_tools' or plugin.get('graph_rpo_credit_backend') != 'outcome_only':
        raise ValueError('Guarded joint RL requires explicit_tools and outcome_only')
    if not plugin.get('verify_training_context') or not plugin.get('graph_reject_repeated_compact'):
        raise ValueError('Guarded joint RL requires context verification and repeated compact protection')
    if any(plugin.get(k, False) for k in ('structured_graph_controller', 'enable_summary',
            'consolidation_interval', 'initial_consolidation_turn', 'auto_prune_max_active',
            'auto_bind_branch_edges', 'enable_retrieval_memory', 'graph_rpo_scope_process_reward')):
        raise ValueError('Guarded joint RL forbids automatic graph maintenance and scope shaping')
