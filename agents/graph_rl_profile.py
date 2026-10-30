"""Model-owned joint RL profiles with verified illegal-action penalties."""

PROFILE = 'invalid_action_v1'


def validate_profile(algorithm, plugin):
    profile = plugin.get('graph_rl_guard_profile', 'none')
    if plugin.get('graph_rpo_credit_backend') == 'management' and profile != 'graph_management_v1':
        raise ValueError('Management credit requires graph_management_v1')
    if profile == 'none':
        return
    management = profile == 'graph_management_v1'
    if profile not in {PROFILE, 'graph_management_v1'}:
        raise ValueError('Unknown graph RL guard profile')
    if (str(algorithm.get('adv_estimator', '')).lower() not in {'graphrpo', 'advantageestimator.graphrpo'}
            or algorithm.get('graphrpo_invalid_action_policy') != 'negative_invalid_graph'
            or bool(algorithm.get('graphrpo_require_binary_reward', True)) != (not management)):
        raise ValueError('Guarded joint RL requires GraphRPO, matching reward type and negative_invalid_graph')
    backend = 'management' if management else 'outcome_only'
    if plugin.get('graph_action_protocol') != 'explicit_tools' or plugin.get('graph_rpo_credit_backend') != backend:
        raise ValueError('Guarded joint RL requires explicit_tools and the matching credit backend')
    if management and (algorithm.get('graphrpo_alpha') != 1. or algorithm.get('graphrpo_beta') != 1.
                       or algorithm.get('graphrpo_norm_adv_by_std', True)):
        raise ValueError('Graph management requires alpha=beta=1 and mean-centered outcome advantages')
    if not plugin.get('verify_training_context') or not plugin.get('graph_reject_repeated_compact'):
        raise ValueError('Guarded joint RL requires context verification and repeated compact protection')
    if any(plugin.get(k, False) for k in ('structured_graph_controller', 'enable_summary',
            'consolidation_interval', 'initial_consolidation_turn', 'auto_prune_max_active',
            'auto_bind_branch_edges', 'enable_retrieval_memory', 'graph_rpo_scope_process_reward')):
        raise ValueError('Guarded joint RL forbids automatic graph maintenance and scope shaping')
