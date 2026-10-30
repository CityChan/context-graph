# Graph size, model-edge density, and effective-operation credit

The guarded RL entry now defaults to `graph_management_v1` with backend
`management`. It restores the structural reward terms requested in the supplied
recipe screenshots while retaining model-owned timing, invalid graph-action
negative advantages, repeated-compact protection, and native-context audits.

## Terminal reward

`R = native_task_reward + 0.2 * G`, with binary native task reward retained
separately for accuracy reporting. Evaluation reports that native reward without
the training bonus. For fewer than four observation nodes, `G = 0`. Otherwise:

`G = 0.4 * archive_ratio + 0.4 * size_score + 0.2 * density_score`

| Component | Exact implementation |
| --- | --- |
| `archive_ratio` | Inactive observation nodes / all observation nodes; both folded and pruned observations count |
| `size_score` | `min(1, 12 / max(active_nodes, 1))`; includes all active node types |
| `density_score` | `min(1, active_model_edges / max(active_nodes - 1, 1))` |
| `active_model_edges` | Distinct `(source, target, relation)` edges added by successful explicit model `add_edge` calls, still active at episode end |

The screenshots specify the 40/40/20 weights, 0.2 terminal coefficient, and
four-observation gate, but not the two normalization formulas. The size target
of 12 and capped edge-to-`(active_nodes - 1)` ratio above are explicit implementation
choices; they are not claimed to reproduce the unavailable old branch exactly.
Framework-created observation, branch, or merge edges do not count as model edges.
This is a structural proxy, not a semantic evidence-quality score. Incorrect
answers can receive up to 0.2 terminal shaping, as in the requested formulation.

## Effective operations and caps

Only successful model operations with the following effects receive local credit:

| Operation | Condition | Raw local credit |
| --- | --- | --- |
| `merge` | Active-node count decreases | +0.1 |
| `prune` | Active-node count decreases | +0.1 |
| `add_edge` | A new edge is created | +0.05 |
| `graph_compact` | At least 2,000 tokens saved and pre-call context <=75% of the working window | +0.2 |

Compact receives at most two bonuses per episode. All positive local credit
combined is capped at 1.0 across main and branches; the final credited operation
may receive only the remaining allowance. Raw call count has no unbounded bonus.
Failed calls, no-ops, read/select/recall calls, and framework operations earn no
positive local credit. Logs distinguish successful, effective, and credited
operation counts. Repeated identical compact is still rejected by the executor.

With `alpha=beta=1` and outcome mean-centering without standard-deviation division:

`A0 = R - mean_group(R) + C + P`

On verified invalid graph-action tokens, `A = min(A0, invalid_penalty)`; all
other tokens use `A0`. The current invalid penalty is -0.3. The raw task result
is still binary, but shaped `R` is not, so this profile explicitly disables the
trainer's binary-total-reward constraint. All views of an episode share the same
terminal `R`; local `C` stays on its sampled action tokens, including after
compaction and batch packing.

The screenshot's token-efficiency reward, forced-finish discount, automatic
maintenance, and broad `negative_all` heuristic shaping are not included in
this change. There are no QA probes or alternating execution/management phases.

## Launch and audit

In an existing four-node Vista allocation, from a clean checkout:

```bash
bash scripts/train_bcp_qwen35_9b_guarded_rl.sh
```

This is a 64K, one-step mechanics smoke (one search node and three trainer ranks).
For the existing five-node, 50-step recipe:

```bash
SMOKE_TEST=0 bash scripts/train_bcp_qwen35_9b_guarded_rl.sh
```

To reproduce the previous binary-outcome guarded profile:

```bash
GRAPH_RPO_GUARD_PROFILE=invalid_action_v1 bash scripts/train_bcp_qwen35_9b_guarded_rl.sh
```

Inspect `training-config.txt` for `graph_management_v1`, backend `management`, and
alpha/beta 1.0. `[GRAPH MANAGEMENT REWARD]` and exported `env_stats` expose each
component, credited operation counts, and the shared cap. Check both raw accuracy
and training reward; higher shaping alone is not improved task success. A short
smoke may contain no graph edits and cannot by itself establish reward coverage.
No live GPU execution or benchmark improvement is claimed by these code tests.
