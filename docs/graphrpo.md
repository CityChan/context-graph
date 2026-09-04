# GraphRPO implementation

This implementation follows Section 4.2 of the ContextGraph manuscript. It is
enabled only when `algorithm.adv_estimator=graphrpo` and
`actor_rollout_ref.actor.policy_loss.loss_mode=graphrpo`.

## Training semantics

- A `gen_uid` identifies one episode. Its main and branch streams share the
  verified binary terminal reward and one episode-level outcome advantage.
- The group baseline uses the population standard deviation. A zero-variance
  group receives zero outcome advantage, and every question must contain at
  least two non-dummy episodes.
- Valid policy-generated `merge`, `prune`, `add_edge`, and `select` decisions
  receive the clipped frozen-evaluator utility increment on successful
  episodes. `pass`, automatic edits, failed edits, and failed episodes receive
  no graph increment.
- Edit increments and non-positive process labels are broadcast across their
  generated assistant-turn spans. Multiple process labels use the most
  negative applicable value.
- The actor loss is weighted first uniformly over all policy tokens in an
  episode, then uniformly over episodes for a question, then over questions in
  the global batch. The same weights are used for entropy and low-variance KL.

GraphRPO deliberately requires the structured graph controller. This makes one
controller response exactly one edit decision span and prevents legacy XML
graph calls from being mixed with environment actions.

## Frozen graph evaluator

`scripts/prepare_graph_evaluator_data.py` extracts question-disjoint binary
training data from saved rollout result JSON. `scripts/train_graph_evaluator.py`
trains a sequence classifier and fits a scalar validation temperature.
`scripts/serve_graph_evaluator.py` loads that checkpoint in evaluation mode and
serves the `contextgraph.graph_evaluator.v1` protocol at `POST /score`.

The policy launcher never substitutes a heuristic score when the evaluator is
missing or unavailable. Set `GRAPH_RPO_EVALUATOR_URL` to the frozen service's
`/score` endpoint before launching
`scripts/train_bc_ctxgraph_8b_graphrpo_5node_48h.sh`.

The evaluator checkpoint and calibration must be fixed before a policy run.
Judge outputs used for task outcomes or scope labels should likewise be cached
or generated with a fixed decoding configuration for reproducible experiments.

Curated SFT Parquet files are not sufficient evaluator data by themselves:
strict SFT builders retain successful demonstrations and therefore omit the
matched negative outcomes. The raw pre-SFT `interactive_results_*.json` and
`gaia_results_*.json` files can be reused because normally completed task
failures retain their graph traces and binary terminal rewards. Infrastructure
failures without a graph trace are ignored. Run
`scripts/pilot_train_graph_rpo_evaluator_from_sft_raw.sh` for a bounded pilot;
because those data come from different tasks and/or teacher policies, the
result must still be fine-tuned and calibrated on held-out target-policy
BrowseComp rollouts before a formal policy experiment.
The pilot treats the raw SFT tree as read-only and writes derived Parquet data
under `$SCRATCH/context-graph-evaluator-data/`; model weights and calibration
are written separately under `$SCRATCH/context-graph-evaluators/`. Path guards
reject either output when it is configured inside the source SFT tree.

## BrowseComp judge audit

BrowseComp task success is positive only after strict answer matching or a
successfully parsed positive LLM judgment. The legacy `relaxed_em` heuristic is
retained as diagnostics but cannot override a negative judgment or create a
positive label when the judge is unavailable.

The mechanics smoke enables `SAVE_ROLLOUT_DATA=1` and writes JSONL records under
`$SCRATCH/context-graph-rollouts/<experiment>/`. Each record includes the task
and episode IDs, binary task reward, graph trace/state, and complete
`judge_audit` decision path. Audit a completed directory with:

```bash
python scripts/audit_bc_judge_results.py /scratch/09281/chc_1996/context-graph-rollouts/train_ctxgraph_EXPERIMENT --fail-on-integrity-error
```

The command deduplicates main/branch streams by `gen_uid`, checks that persisted
task rewards match judge decisions, reports grader parse failures, and prints
all non-strict positive decisions for manual review. W&B reward, graph, and
judge metrics are also episode-weighted rather than branch-stream-weighted.

For mechanics-only validation on an existing four- or five-node allocation,
`scripts/smoke_train_bc_ctxgraph_8b_graphrpo_5node_idev.sh` starts a deterministic
CPU evaluator on the search node and performs one optimizer step. Its evaluator
scores are deliberately synthetic: use the smoke only to verify wiring and
never include its reward or checkpoint in experiments.

On an existing four-node allocation, run the original Qwen3-8B zero-shot
judge-audit variant, including rollout persistence and post-run audit, with:

```bash
bash scripts/smoke_train_bc_ctxgraph_8b_graphrpo_qwen3_8b_4node_judge_audit.sh
```
