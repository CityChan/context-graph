# GraphRPO training hardening

These are new implementations based on the other device's change description,
not a reconstruction of its unavailable patch. Existing inference defaults stay
unchanged. GPU/ROCm performance and distributed checkpoint recovery still need
live validation; CPU tests do not establish training effectiveness.

## Training

Optional Hydra settings (all new switches default off):

| Setting | Behavior |
| --- | --- |
| `algorithm.graphrpo_dynamic_sampling=true` | Keep whole informative groups and request more rollouts before any update. |
| `algorithm.graphrpo_dynamic_max_gen_batches=4` | Maximum generation batches per update attempt. |
| `algorithm.graphrpo_dynamic_min_groups=1` | Stop once this many useful groups are available; retain at most `data.train_batch_size`. |
| `algorithm.foldgrpo_dynamic_*` | Equivalent bounded sampling for FoldGRPO, using deduplicated episode outcome variance. |
| `algorithm.graphrpo_decision_scale_max=32` | Cap local graph-credit amplification when decision-token normalization is enabled; `0` means uncapped. |
| `plugin.process_reward=[compress]` | Optional heuristic cost penalties for ordinary GraphRPO. |

`plugin` abbreviates `actor_rollout_ref.rollout.plugin` above.
The existing Vista training launcher exposes `GRAPH_RPO_DYNAMIC_SAMPLING`,
`GRAPH_RPO_DYNAMIC_MAX_GEN_BATCHES`, `GRAPH_RPO_DYNAMIC_MIN_GROUPS`,
`GRAPH_RPO_DECISION_SCALE_MAX`.

Joint GraphRPO retains groups with outcome diversity and also retains ties with
negative process labels (or opt-in graph credit) whose coefficients are enabled.
All streams of a candidate stay together. Rejected rollout tokens do not count.
Empty selection skips the optimizer and scheduler. Bounded sampling can return
fewer than the requested minimum; telemetry reports this instead of retrying
forever. It selects a conditional training distribution: report generation cost
and kept/seen groups when comparing experiments.

Additional batches consume the same epoch iterator. Dynamic sampling can reduce
updates per epoch. Checkpoints record epoch independently of update count in
`trainer_progress.json`; dynamic resume rejects old checkpoints without this
metadata. Epoch exhaustion saves the final state when checkpointing is enabled.
Scope process-reward calls select `SCOPE_JUDGE_MODEL`, then `JUDGE_MODEL`, then
the existing `gpt-5-nano` default. Ungraded responses are logged and remain neutral;
this selects a model, not a new authentication or Papyrus transport backend.
The trainer's validation groups normalize array/list identifiers before grouping,
preserving composite IDs and merging scalar wrappers with their scalar IDs.

NaNs on masked tokens are removed *before* loss arithmetic. NaNs on active tokens
produce an explicit metric and nonfinite loss, retaining the existing all-rank
finite-loss guard. We do not silently zero corrupt gradients. Dummy masks and
credit removed by rollout rejection are cleared.

## Optional compression process labels

The labels apply to generated assistant tokens only, with total per-turn penalty
bounded below by -1. Existing stronger penalties retain precedence in GraphRPO.

| Label | Trigger | Penalty |
| --- | --- | --- |
| `trunc` | Reported output limit, or exact token cap reached without EOS | -1 |
| `dup_search` | Query word-set Jaccard >= .8, or repeated page identifier | -.3 |
| `dup_branch` | Branch prompt Jaccard >= .5 / branch limit exceeded | -.5 / -1 |
| `redund` | After half the context budget, >300-token response with >.5 previously visible 8-gram overlap | -.5 times overlap |
| `ret_redund` | Return >1500 tokens or similar to a previous branch return | -.3 |

These are heuristics, not evidence that an action is incorrect. Queries/pages and
visible context are tracked per stream, branch returns across streams. Report as
a separate reward ablation. This adds no new environment or graph action.

## ROCm allocation controls

- `VERL_PAD_TRIM_BUCKET=2048`: opt-in text-only rectangular padding trim. Remove
  only columns masked for every row, keep a prompt predecessor, retain position
  IDs, and pad output log-probabilities back to the original response width.
  Disabled for multimodal inputs and the existing remove-padding path.
- `VERL_FRAG_EMPTY_CACHE_GB=24`: release unused allocator blocks after forward
  scoring/backward if reserved-minus-allocated exceeds the threshold. Default 0
  disables it. This does not free live tensors or guarantee OOM avoidance.
- Peak allocation statistics reset per actor update when these controls are used.
