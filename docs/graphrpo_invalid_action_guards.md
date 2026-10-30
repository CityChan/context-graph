# Joint GraphRPO with invalid-action guards

This page describes the preserved `invalid_action_v1` profile. The guarded entry
now defaults to [graph_management_full_v1](graphrpo_full_alignment.md), which adds
structural and process shaping, `negative_all`, and optional automatic maintenance. Select
`GRAPH_RPO_GUARD_PROFILE=invalid_action_v1` to reproduce this page's binary-outcome
recipe.

`invalid_action_v1` implements the joint execution/graph-management design: one
policy chooses task actions and explicit graph tools, including when to compact.
The final task reward is binary. The profile has no automatic prune, compact,
periodic controller, scope shaping, or structural graph bonuses. Existing
launchers retain their previous behavior unless this profile is selected.

## Advantage and action provenance

The existing group outcome advantage, graph credit, and process label computation
remains `A0 = group_advantage(R) + alpha*C + beta*P`. In this profile the
`outcome_only` backend gives no separate positive graph credit. For tokens of a
graph call rejected by the executor, a separate provenance mask stores the raw
invalid-action penalty (currently -0.3), and the final advantage is
`A = min(A0, invalid_penalty)`. All other policy tokens retain `A0`.

Thus a correct answer cannot turn an invalid graph call into a positive policy
update. This is `negative_invalid_graph`, not the screenshot's `negative_all`:
heuristic scope labels or repeated searches are not automatically classified as
invalid graph actions. Malformed action text that cannot be reliably attributed
to a graph tool is also not relabeled as one. The existing action protocol handles
such errors. Task execution and valid graph management still learn jointly from
the outcome advantage.

The negative mask is carried through segments, padding, Ray batches, dummy rows,
and dynamic group selection. Tied-outcome groups with active invalid-action
penalties remain useful even if the ordinary process coefficient is zero.
Missing or inconsistent provenance fails validation.

## Compaction and training context

Repeated `graph_compact` is rejected when its retained prompt is identical to
the last successful compact in the same agent. A changed graph, focus, or retained
task prefix can permit another compact. Failed calls do not update the fingerprint.
This guard does not assess semantic summary quality or penalize legitimate search
retries. No framework action is represented as a sampled policy action.

With `verify_training_context=True`, each sampled call records fingerprints of
its native input IDs, output IDs, and rollout logprobs. Export verifies these
against the actual segmented training tensors and checks token masks and credit
positions. Backend input IDs are compared when supplied. Changes or lost sampled
tokens cause an error rather than a silent update on a different prefix.

Normal compatible segments stay combined. If a compacted prefix or segment
exceeds the fixed prompt/response allocation, export falls back to native request
rows. Prefix overflow occupies masked response slots; each row supervises only
that request's sampled output. Historical outputs and observations receive zero
loss and zero credit. Requests that cannot fit the full allocation fail closed.
This can increase the number of rows and training cost; candidate identity and
group reward accounting remain unchanged. Verification covers saved rollout
logprobs, not numerical equality with GPU-recomputed old-policy logprobs.

## Vista entry point

From a clean checkout of this branch inside a four-node allocation:

```bash
GRAPH_RPO_GUARD_PROFILE=invalid_action_v1 bash scripts/train_bcp_qwen35_9b_guarded_rl.sh
```

This defaults to the existing Qwen3.5-9B checkpoint, 65,536 context tokens
(8,192 prompt + 57,344 response allocation), three training ranks plus one search
node, three prompts with two rollouts each, and one optimizer step. Its short
four-turn rollout is a mechanics smoke, not a performance evaluation. The model
may not choose a graph edit within that budget; a successful job alone does not
prove that invalid-action or compaction guards were exercised.

Inside a five-node allocation, the existing 50-step recipe can be selected with:

```bash
GRAPH_RPO_GUARD_PROFILE=invalid_action_v1 SMOKE_TEST=0 bash scripts/train_bcp_qwen35_9b_guarded_rl.sh
```

The final configuration is recorded in `training-config.txt`; checkpoint and
rollout audits remain enabled for the smoke. The profile is checked again at
rollout startup to reject incompatible maintenance/shaping settings. These
commands run within an existing allocation and do not submit a new Slurm job.

This change does not copy the screenshot's terminal graph score, operation-count
bonuses, automatic maintenance, or all heuristic penalties. It also does not
change the SFT distillation recipe. Local tests cover executor rejection,
advantage signs, exact-context packing, batch propagation, dynamic sampling, and
launcher configuration. Live 64K GPU training and benchmark gains remain to be
validated separately.
