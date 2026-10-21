#!/bin/bash
# Resolve the saved training protocol before constructing Hydra arguments.
if [ -z "${BC_CTXGRAPH_PROTOCOL:-}" ]; then
  if [ "${ADV_ESTIMATOR:-foldgrpo}" = graphrpo ]; then
    BC_CTXGRAPH_PROTOCOL=explicit_tools
  else
    BC_CTXGRAPH_PROTOCOL=legacy
  fi
fi
GRAPH_PROTOCOL_ARGS=()
BC_AUTO_BIND_BRANCH_EDGES=True
case "$BC_CTXGRAPH_PROTOCOL" in
  legacy|full_policy)
    BC_STRUCTURED_GRAPH_CONTROLLER=false
    BC_CONTROLLER_OWNED_TOOL_FORMATTING=false
    ;;
  controller)
    BC_STRUCTURED_GRAPH_CONTROLLER=true
    BC_CONTROLLER_OWNED_TOOL_FORMATTING=true
    ;;
  explicit_tools)
    BC_STRUCTURED_GRAPH_CONTROLLER=false
    BC_CONTROLLER_OWNED_TOOL_FORMATTING=false
    CONSOLIDATION_INTERVAL=0
    AUTO_PRUNE_MAX_ACTIVE=0
    BC_AUTO_BIND_BRANCH_EDGES=False
    GRAPH_PROTOCOL_ARGS=(
      +actor_rollout_ref.rollout.plugin.graph_action_protocol=explicit_tools
      +actor_rollout_ref.rollout.plugin.action_output_profile=typed_action_json_v1
      +actor_rollout_ref.rollout.plugin.graph_decision_profile=graph_affordances_v2
      +actor_rollout_ref.rollout.plugin.initial_consolidation_turn=0
      +actor_rollout_ref.rollout.plugin.enable_retrieval_memory=False
      +actor_rollout_ref.rollout.plugin.inject_graph_state_after_action=True
      '+actor_rollout_ref.rollout.plugin.apply_chat_template_kwargs={enable_thinking:false,preserve_thinking:true}'
    )
    ;;
  *)
    echo "ERROR: BC_CTXGRAPH_PROTOCOL must be explicit_tools, controller, full_policy, or legacy" >&2
    exit 2
    ;;
esac
