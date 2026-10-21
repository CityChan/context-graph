#!/bin/bash
# Source before setup so removed options cannot silently select a new objective.
for graph_rpo_option in $(compgen -A variable GRAPH_RPO_); do
  case "$graph_rpo_option" in
    GRAPH_RPO_COUNTERFACTUAL_*|GRAPH_RPO_CONTINUATION_*|GRAPH_RPO_MEMORY_ONLY|GRAPH_RPO_ALTERNATING_ROLES|GRAPH_RPO_EXECUTOR_SAMPLES|GRAPH_RPO_TIMEOUT_AS_FAILURE)
      echo "ERROR: removed GraphRPO option $graph_rpo_option; use joint execution/graph training" >&2
      exit 2
      ;;
  esac
done
case "${GRAPH_RPO_CREDIT_BACKEND:-outcome_only}" in
  old_policy_counterfactual_qa|old_policy_continuation)
    echo "ERROR: removed GraphRPO backend $GRAPH_RPO_CREDIT_BACKEND; use outcome_only" >&2
    exit 2
    ;;
esac
unset graph_rpo_option
