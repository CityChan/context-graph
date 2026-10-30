"""Model-owned joint RL profiles with verified illegal-action penalties."""

PROFILE = 'invalid_action_v1'


def validate_profile(algorithm, plugin):
    profile = plugin.get('graph_rl_guard_profile', 'none')
    full = profile == 'graph_management_full_v1'
    if plugin.get('graph_rpo_credit_backend') == 'management' and profile not in {'graph_management_v1', 'graph_management_full_v1'}:
        raise ValueError('Management credit requires a graph management profile')
    if profile == 'none':
        return
    management = profile == 'graph_management_v1' or full
    if profile not in {PROFILE, 'graph_management_v1', 'graph_management_full_v1'}:
        raise ValueError('Unknown graph RL guard profile')
    if (str(algorithm.get('adv_estimator', '')).lower() not in {'graphrpo', 'advantageestimator.graphrpo'}
            or algorithm.get('graphrpo_invalid_action_policy') != ('negative_all' if full else 'negative_invalid_graph')
            or bool(algorithm.get('graphrpo_require_binary_reward', True)) != (not management)):
        raise ValueError('Guarded joint RL requires GraphRPO with the profile-matched reward and negative advantage policy')
    backend = 'management' if management else 'outcome_only'
    if plugin.get('graph_action_protocol') != 'explicit_tools' or plugin.get('graph_rpo_credit_backend') != backend:
        raise ValueError('Guarded joint RL requires explicit_tools and the matching credit backend')
    if management and (algorithm.get('graphrpo_alpha') != 1. or algorithm.get('graphrpo_beta') != 1.
                       or algorithm.get('graphrpo_norm_adv_by_std', True)):
        raise ValueError('Graph management requires alpha=beta=1 and mean-centered outcome advantages')
    if not plugin.get('verify_training_context') or not plugin.get('graph_reject_repeated_compact'):
        raise ValueError('Guarded joint RL requires context verification and repeated compact protection')
    if full:
        if not 0 < float(plugin.get('gm_auto_compact_threshold', .75)) <= 1:
            raise ValueError('Invalid auto compact threshold')
        if not 0 < float(plugin.get('gm_auto_compact_min_gain', .25)) < 1:
            raise ValueError('Invalid auto compact minimum gain')
        if not 0 < float(plugin.get('gm_branch_similarity_threshold', .8)) <= 1:
            raise ValueError('Invalid branch similarity threshold')
        if any(int(plugin.get(key, 0)) < 0 for key in ('gm_auto_prune_max_active',
                'gm_auto_compact_max', 'gm_auto_compact_cooldown', 'graph_compact_node_chars')):
            raise ValueError('Negative graph management limit')
    if any(plugin.get(k, False) for k in ('structured_graph_controller', 'enable_summary',
            'consolidation_interval', 'initial_consolidation_turn', 'auto_prune_max_active',
            'auto_bind_branch_edges', 'enable_retrieval_memory', 'graph_rpo_scope_process_reward')):
        raise ValueError('Guarded joint RL forbids legacy automatic graph maintenance and scope shaping; full profile uses gm_auto_*')
