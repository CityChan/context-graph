# Simplified prune/search RL (`prune_search_v1`)

This is an opt-in alternative to the existing graph management recipes. It trains
task execution and model-selected graph operations together with explicit JSON
actions plus thinking. Existing launchers retain their previous default profiles.

## Objective

- Terminal outcome `R` is correct = 1, wrong = 0. Forced and natural answers use
  the same score. There is no graph-size, edge-density, efficiency or action-count
  terminal reward.
- A model prune that succeeds and reduces active nodes is eligible for `C = 0.1`
  on that sampled action, **only if the whole episode ultimately answers correctly**.
  Main and branches share an episode cap of 0.5. A recalled node loses eligibility;
  repeated prune/recall cycles cannot earn credit. At settlement the node must
  still be pruned. Eligible actions are paid in execution order up to the cap.
- Duplicate successful searches and executor-rejected actions receive `P = -0.3`.
  This includes malformed actions and invalid/nonoperable graph IDs.
- `A0 = R - mean_group(R) + C + P`, with alpha = beta = 1 and no standard-deviation
  normalization. `negative_all` makes each negatively labeled action's advantage
  `min(A0, -0.3)`. Final correctness cannot turn its penalty into positive advantage.

Credits and penalties use the existing sampled-action masks; the terminal reward
remains binary. Failed episodes with local learning signals are retained. The
old unfolded-main, no-finish, scope and graph-management shaping paths are skipped.
Merge, edge creation and compact receive no positive credit in this profile.

## Executor behavior in training and evaluation

Search deduplication uses case-folded, whitespace-normalized query plus effective
display top-k (default and cap applied as in LocalSearch). The successful-search
registry is shared across the main agent and branches, and survives prune and
compact. Duplicate requests are rejected **before contacting the search service**;
they consume a turn and produce feedback, but no evidence node. Failed retrievals
remain retryable. A successful search yielding no newly visited documents gives
feedback without an extra penalty or evidence node; that successful key is cached.

Automatic prune, compact, controller scheduling, history replacement and scope
shaping are disabled. The model chooses graph operations and compact timing.
Existing repeated-compact rejection and exact sampled-context/token/logprob/mask
verification remain enabled. Compact savings are recorded from actual before/after
token counts as `prune_search_compact_tokens_saved`, not rewarded.

## Entry points

On an existing four-node Vista allocation, the opt-in launcher selects a 64K,
one-step mechanics smoke:

```bash
bash scripts/train_bcp_qwen35_9b_prune_search_rl.sh
```

`SMOKE_TEST=0` selects the existing five-node full training launcher. It does not
submit a scheduler job by itself. Inspect the saved effective config and use a
matching allocation before selecting full training.

Standalone evaluation uses `scripts/eval_bcp_qwen38.py` with
`--method contextgraph --graph-action-protocol explicit_tools --graph-rl-profile prune_search_v1`
in addition to its usual data, endpoint, model and output arguments. This selects
64K working context, thinking, the same executor restrictions and training-default
retrieval limits. Other evaluation protocols remain unchanged. Sampling temperature
and turn/time limits must still be matched explicitly for a controlled comparison.

## Validation scope

CPU tests exercise the real LocalSearch executor with mocked retrieval/model calls,
main/branch sharing, reward gating and cap, recall cancellation, failed-search
retries, malformed actions, forced answers, compact and duplicate rejection.
Launcher tests compose real Hydra overrides. These checks do not demonstrate GPU
training, convergence, semantic pruning correctness or benchmark improvement.
