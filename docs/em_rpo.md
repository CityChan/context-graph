# EM-RPO: executor and memory training

EM-RPO is the name of the alternating E+M training variant previously called
E+M GraphRPO. One model and optimizer serve both roles, using the existing
executor prompt and structured graph-maintenance requests. The earlier M-only
and proxy-credit GraphRPO variants remain separate experiments.

## Training procedure

- **E update:** sample K complete episodes for each question. Compute terminal
  leave-one-out advantage, `R_i - mean(R_j, j != i)`, and train execution,
  branch/return and answer tokens. Exclude every maintenance completion,
  including pass and rejected edits. Main and branch streams count as one
  episode in the reward baseline.
- **M update:** record a prefix up to a selected maintenance checkpoint, replay
  that same state for K sampled maintenance decisions, and execute each real
  continuation. Train only the selected maintenance completion, using the
  continuation group's leave-one-out terminal advantage.

Odd saved rollout-attempt steps select E and even steps select M. Skipped
attempts also advance this schedule; two attempts do not guarantee two optimizer
updates. Each update applies PPO clipping and KL only on its selected role's
tokens. The other role has no direct loss, but shared parameter updates can
change both roles' behavior. Service failures exclude the whole comparison
group. Equal rewards give zero task advantage, although KL may still contribute.

## Vista pilot

From the repository root, submit:

```bash
sbatch scripts/train_bcp_em_rpo.sh
```

The pilot requests five nodes for 24 hours and defaults to two rollout attempts
(E then M), four questions per attempt, K=2 per role and concurrency 2 per group.
It reuses the existing model/data setup. `GRAPH_RPO_EXECUTOR_SAMPLES` and
`GRAPH_RPO_CONTINUATION_SAMPLES` set the two group sizes.

## Compatibility

This rename adds a canonical launcher and documentation; it does not change
training mathematics, inference, or benchmark protocols. The old
`scripts/train_bcp_graphrpo_em.sh` delegates to the new entry point and retains
its previous default run tag and Slurm log names.

Persisted names stay unchanged so existing overrides and analysis scripts work:

- `algorithm.adv_estimator=graphrpo` and policy loss mode `graphrpo`.
- `algorithm.graphrpo_alternating_roles=True` and
  `algorithm.graphrpo_memory_only=False`.
- Credit backend `old_policy_continuation` and `GRAPH_RPO_*` environment variables.
- Per-row `graph_rpo_training_role` and metrics `graphrpo/E/*`, `graphrpo/M/*`.

See [GraphRPO implementation details](graphrpo.md) for replay checks, credit
normalization and the older experimental backends. Resuming an evaluation still
requires its original pinned commit; naming compatibility does not bypass its
manifest checks.

## Implementation limits and verification

The current continuation backend supports only LocalSearch. M trains one
checkpoint per source rollout and runs K continuations; independent executor
sampling adds outcome variance. Binary terminal rewards remain sparse.

Training preserves immutable history for token/log-probability alignment, while
retrieval-memory inference can replace history. This difference in available
memory remains unresolved: successful role masking alone does not establish
that maintenance learns the memory behavior used during evaluation.

Local tests cover agent-loop role masks, branch identity, same-state M replay,
failure exclusion, resumed role selection and both roles updating a shared
optimizer with synthetic token log-probabilities. GPU/FSDP training, checkpoint
reload and held-out improvements still require live validation. Before a long
run, inspect actual E/M optimizer updates, nonzero advantages, skipped groups,
trainable tokens and continuation cost in a short GPU pilot.
