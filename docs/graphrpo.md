# GraphRPO implementation

This implementation follows Section 4.2 of the ContextGraph manuscript. It is
enabled only when `algorithm.adv_estimator=graphrpo` and
`actor_rollout_ref.actor.policy_loss.loss_mode=graphrpo`.

## Alternating executor and memory training (E+M, opt-in)

Set `algorithm.graphrpo_alternating_roles=True`,
`algorithm.graphrpo_memory_only=False` and keep the
`old_policy_continuation` backend. The same actor and optimizer train both
roles. Odd saved training steps train E; even steps train M. The step counts
rollout attempts, including skipped batches, so resuming does not reset the
schedule or repeatedly retry an unavailable role.

- **E:** sample K independent complete episodes of the same question with the
  pre-update shared policy. Give each episode leave-one-out terminal advantage
  `R_i - mean(R_j, j != i)`. Train its execution tokens, including branch/return
  and answer tokens; exclude every controller completion, including pass and
  rejected edits. Main and branch streams share an episode ID and count once
  in the reward baseline. Weight questions, episodes, then all E tokens in an
  episode equally at their respective levels. No graph or process bonus is added.
- **M:** use the same-state continuation procedure below. Only the selected
  maintenance completion trains, using its continuation group's relative reward.

Each role retains PPO clipping and KL on its selected tokens. E and M use the
existing executor prompts and structured maintenance requests; this does not
introduce a second model, adapter, inference loop or automatic evaluator change.
The other role participates in rollouts but has no direct loss in that update.
Shared weights change after each update, so neither role is permanently frozen.
Terminal reward estimates remain noisy and all-tied groups have zero task credit.

Both roles use whole-group service-failure exclusion and the actor's collective
finite-loss guard. `graph_rpo_training_role` is saved per row; the trainer checks
it against the saved step before applying credit. `graphrpo/E/*` and
`graphrpo/M/*` report candidate counts, skipped groups, trainable tokens, nonzero
advantage fractions and actor update metrics separately.

The Vista pilot defaults to two rollout attempts (E then M), four questions per
attempt, K=2 for both roles and per-group concurrency 2. It uses five nodes and
a 24-hour allocation, reusing the existing model/data setup:

```bash
sbatch scripts/train_bcp_graphrpo_em.sh
```

`GRAPH_RPO_EXECUTOR_SAMPLES` and `GRAPH_RPO_CONTINUATION_SAMPLES` control the two
group sizes. The old M-only launcher/defaults remain available. This E+M path
still supports only LocalSearch and still uses immutable training history; it
does not fix the history-replacement difference from retrieval-memory inference.
Tests cover real agent-loop role masks and branches, driver credit/normalization,
and both roles updating one optimizer with synthetic token log-probabilities.
They do not establish successful GPU training or task improvement.

## Same-state continuation experiment (M-only, opt-in)

`graph_rpo_credit_backend=old_policy_continuation` with
`algorithm.graphrpo_memory_only=True` trains the existing maintenance role of
the shared model. Inference prompts, controller scheduling, legal actions and
all previous backend defaults stay unchanged. This is **real tool continuation**,
not the older tool-free before/after QA probe below.

One ordinary rollout records an ordered tape of model completions and raw
LocalSearch RPC results up to a maintenance checkpoint. Each candidate replays
that prefix through the **same agent loop**, including branches, archives,
visited-document state, graph operations and token/turn accounting. Replayed
requests must match, and the checkpoint verifies a hash of the actual model
input, graph/archives, environment state and turn count. Prefix wall time is
also charged to every continuation. Replay makes no model/search RPCs and
returns fresh copies of cached results. It does not deep-copy live connections
or pretend a graph snapshot restores an arbitrary simulator.

At that checkpoint, K independently sampled M decisions each continue with the
pre-update model using real tools and the normal task judge. Later maintenance
uses the same frozen policy and schedule; it is not recursively branched. For
candidate i, its credit is `R_i - mean(R_j for j != i)`. Ties produce zero task
advantage; duplicates, legal passes and rejected decisions are not selectively
discarded. There is no compression bonus, operation cost, outcome normalization
or extra episode/process reward in this mode. PPO's existing KL regularizer
still applies on selected M tokens. The pilot also enables the existing clipped
token importance correction for rollout/training probability differences.

Only the selected M completion is trainable. The recorded prefix is retained as
conditioning tokens with a zero loss mask; future E/M tokens and branch streams
are excluded from training. The emitted M prefix is checked against the exact
model input, and original token IDs/logprobs are retained. Each maintenance
state has a fresh group ID and each candidate a distinct episode ID, so the
existing GraphRPO loss averages states, candidates, then decision tokens.
An episode ending before the chosen checkpoint emits a zero-mask sample and is
excluded from loss normalization. Exhausted network/timeout failures, HTTP
408/429/5xx responses and explicit environment/judge failure flags discard the
whole group and emit one zero-mask audit sample. Completed siblings are not
salvaged: doing so would change the comparison group. Queued siblings stop and
running siblings are cancelled and drained before returning. Replay/token
inconsistency, configuration/authentication errors and unknown programming errors
still propagate; caller cancellation is not converted into a training sample.

An entirely zero-mask batch skips model computation, optimizer and scheduler
updates and empty-tensor metrics, while saving the rollout audit. It consumes a
rollout attempt in the existing step budget, reported by
`training/skipped_empty_memory_batch=1` and `training/actor_update_calls=0`.
Validation/checkpoint hooks for that skipped attempt are not run. Valid tied
groups retain their M masks and can still receive KL regularization.

The shared FSDP actor checks policy, optional KL and total losses collectively
before each microbatch backward. If any rank reports a non-finite loss, every
rank clears accumulated gradients and raises `FloatingPointError` before that
backward or optimizer step. This deliberately stops training for diagnosis;
it neither substitutes a zero reward nor attempts recovery with a zero loss.
Finite zero-mask microbatches still participate in backward. The guard applies
to all objectives using this actor, and adds one collective per microbatch.
CPU/Gloo tests cover asymmetric NaN/Inf failures and a zero-mask rank using the
actual actor update loop; GPU FSDP/NCCL execution remains a separate smoke test.

Limits of this first implementation:

- Only the read-only `LocalSearch` environment is supported. No SWE, ScienceWorld,
  or DiscoveryWorld cloning is implemented. Session restarts and structured fact
  memory are rejected in this training mode.
- One checkpoint per source rollout, configured by the 1-based
  `graph_rpo_continuation_checkpoint` (default 1). `random` samples an ordinal
  uniformly from 1 through `graph_rpo_continuation_checkpoint_max` (default 4)
  **before** executing the prefix, without looking at future rewards. The seed
  and chosen ordinal are saved. Short episodes may not reach it and are skipped;
  this is not uniform sampling over the maintenance points actually reached.
  K is
  `graph_rpo_continuation_samples` (default 4). Controller temperature must be
  positive. `rollout.n` controls source prefixes and can be 1.
- `graph_rpo_continuation_concurrency` (default 2) bounds simultaneous candidates
  **per source group**. Existing source groups/workers remain parallel, so the
  cluster-wide limit also depends on their count. This does not reduce the total
  continuation work or guarantee a particular speedup.
- One continuation per candidate is a noisy estimate. All-failure groups still
  have zero task advantage. There is no automatic curriculum or learned PRM.
- This preserves the existing training context protocol, including its immutable
  history. It does not claim the graph is the executor's only evidence source.
- It assumes the synchronous PPO training cycle: complete the rollout batch
  before updating shared weights. E can change after each optimizer update.

The `graph_rpo_continuation` record in saved training rollouts contains the state
hash, group, checkpoint, policy step, seeds, all candidate rewards and local
advantage. Telemetry includes failed/skipped/nonzero fractions, mean absolute advantage
and decision-token counts. Ordinary validation does not fork and measures the
complete shared-model agent.

On Vista, after setting any model/data overrides required by the existing BC-P
launcher, submit the two-attempt pilot (five nodes, 24-hour allocation):

```bash
sbatch scripts/train_bcp_graphrpo_continuation.sh
```

The wrapper defaults to four source prefixes, four candidates each, concurrency
two per group, random checkpoint ordinals 1..4, four validation examples and
checkpointing every non-skipped step. It reuses the existing SFT
checkpoint/model setup. Increase `TOTAL_TRAINING_STEPS` only after inspecting
replay integrity, nonzero advantages, optimizer metrics and saved checkpoints.
Local tests exercise the real loop with controlled model/search responses;
they are not evidence of successful Vista GPU training or task improvement.

## Training semantics

These are the original objective defaults. The opt-in evidence experiment below
adds branch credit and a separate normalization for decision tokens.

- A `gen_uid` identifies one episode. Its main and branch streams share the
  verified binary terminal reward and one episode-level outcome advantage.
- The group baseline uses the population standard deviation. A zero-variance
  group receives zero outcome advantage, and every question must contain at
  least two non-dummy episodes.
- Valid policy-generated `merge`, `prune`, `add_edge`, and `select` decisions
  receive a clipped graph-utility increment. `pass`, automatic edits, and
  failed edits receive no graph increment. The formal counterfactual backend
  evaluates edits from both successful and unsuccessful main episodes; the
  older evaluator and answer-likelihood backends retain their historical
  terminal-success gate.
- Edit increments and non-positive process labels are broadcast across their
  generated assistant-turn spans. Multiple process labels use the most
  negative applicable value.
- The actor loss is weighted first uniformly over all policy tokens in an
  episode, then uniformly over episodes for a question, then over questions in
  the global batch. The same weights are used for entropy and low-variance KL.

GraphRPO deliberately requires the structured graph controller. This makes one
controller response exactly one edit decision span and prevents legacy XML
graph calls from being mixed with environment actions.

## Evidence and branch-credit experiment

Set `GRAPH_RPO_CREDIT_BACKEND=evidence` for the new BC-P experiment. The default
remains `old_policy_counterfactual_qa`, so existing launch commands keep their
original objective. Evidence credit is supervised by training document annotations;
it is not an evaluator-free correctness estimate or a demonstrated score improvement.

- A branch receives `new_gold_documents / total_gold_documents` on its initiating
  main-agent decision. Documents already retrieved by either the main or a branch
  cannot earn discovery credit again. IDs come from actual displayed search/open
  tool results, never model-written citations or unseen search hits.
- For graph edits, freeze a bounded, question-conditioned retrieval view before
  and after the edit. Credit is `(newly_visible_gold - lost_gold) / total_gold`.
  Visibility requires eight consecutive words from previously tool-returned text
  in the view. A document ID alone earns nothing. This conservative lexical proxy
  can miss paraphrases and is not a fact-entailment judgment. The probe is not a
  replay of the complete main-agent prompt. Archived evidence is not necessarily
  visible; pruning that leaves retrieved evidence available incurs no loss.
- A branch with task-word Jaccard similarity at least 0.6 to a previous branch
  gets a 0.2 penalty only if it retrieved no new document or expanded source passage.
  Opening more text from a previously seen page is exempt (new source eight-word
  shingles), even though it earns no first-discovery bonus. This heuristic is
  audited, not a hard branch ban.
- A bounded history of six branch attempts, their new-document counts and their
  unverified returned reports is shown to the policy. It contains no gold labels
  and does not assert that a no-result search disproved a hypothesis.
- Credit is assigned even when the final answer is wrong. Automatic edits and
  rejected operations get no evidence credit. Valid passes have zero credit and
  count in the decision-token denominator. There is no graph-size penalty.

The evidence launcher enables `graphrpo_normalize_decision_tokens`: for each episode,
the local-credit component is multiplied by total policy tokens across all streams
divided by valid decision tokens. Combined with the existing episode-token loss
weights, this averages local credit over decision tokens, rather than diluting it
over the whole main/branch transcript. Outcome/process terms, entropy and KL retain
their old weights. Zero-credit decisions count; episodes without decisions get zero
local contribution. Signed credit is not centered within a question, which would
erase uniformly useful/harmful decisions. This is a different objective and must
be reported as such.

The Qwen3.5 evidence profile uses alpha 0.5, delta clip 1.0, balanced controller
actions, controller training temperature 0.8 (evaluation stays greedy), and token
importance correction capped at 2.0. It keeps KL enabled and does not increase the
hardware-dependent batch size or drop all-correct/all-wrong groups: those groups
can still contain nonzero evidence credit. Scope-judge shaping is disabled in this
profile. Final answer grading still uses the normal task judge.

### Training labels and four-node smoke

The input training parquet must contain nonempty `extra_info.graph_rpo_gold_docids`
for every row. Labels must identify documents in the retriever's corpus and must
come from training annotations. Missing/malformed labels fail preflight; they are
never silently treated as an empty gold set. Validation does not need or read them.
If existing parquet lacks the field, prepare a new file using JSON mapping exact
training questions to lists of document IDs:

```bash
python scripts/prepare_graph_evidence_data.py --source data/bc_train.parquet --labels /path/to/training-question-docids.json --output /scratch/path/bc_train_evidence.parquet
```

The converter preserves prompts, answers and row ordering, and refuses overwrite.
It does not infer annotation columns, download labels, or modify test data. A labels
file/source path for the real Vista dataset must be supplied before this experiment
can run; the repository does not contain those training annotations.

Inside a free four-node Vista idev allocation, from the committed checkout:

```bash
GRAPH_RPO_TRAIN_DATA=/scratch/path/bc_train_evidence.parquet bash scripts/smoke_bcp_graphrpo_evidence_4node_idev.sh
```

This uses search plus three trainer GPUs, three prompts with two rollouts each,
one optimizer step, a 12K context, up to 16 turns/three branches, and a controller
checkpoint every two main turns. It preserves the existing environment separation.
Completion requires checkpoint shards, judge integrity, persisted evidence decisions
and nonzero local credit. Zero evidence signal is an inconclusive smoke that exits
nonzero, not a successful training validation. It does not establish performance.

Metrics include `graph_rpo_evidence_recall`, `graph_rpo_evidence_gained/lost`,
`graph_rpo_branch_decisions`, `graph_rpo_duplicate_branch_rate`,
`graph_rpo_decision_tokens`, and the existing signed/absolute credit sums. Per-event
traces retain decision indices, bounded views, document sets and credit components.
`scripts/audit_evidence_graph_credit.py` checks saved credit arithmetic and deduplicates
episode streams. It does not revalidate document annotations against the corpus.
Evaluation of a trained evidence-profile policy should also enable
`plugin.graph_branch_history=True`; gold annotations remain unnecessary.

## Formal evaluator-free utility: paired old-policy QA outcomes

`GRAPH_RPO_CREDIT_BACKEND=old_policy_counterfactual_qa` is the default formal
backend. After an episode has produced its graph-edit trace, but before the
actor optimizer step, the same fixed pre-update policy receives a tool-free QA
prompt containing either the graph immediately before an edit or the graph
immediately after it. Both sides use matched sampling seeds. Their independently
generated answers are scored by the environment's ordinary task judge, and
the edit utility is

`mean(R_after) - mean(R_before) - operation_cost`,

clipped by `GRAPH_RPO_DELTA_MAX`. Set
`GRAPH_RPO_COUNTERFACTUAL_SAMPLES` to the number of downstream answers sampled
per graph state. Probe generations and their judge audits are stored in the
graph trace for replay, but probe tokens are not emitted as policy-training
trajectories.

This estimates how useful the graph state is to the policy that actually
consumes it. It does not teacher-force the benchmark answer and does not use
the frozen reference model as a value estimator. The reference model remains
only in the low-variance KL regularizer. Because failed main trajectories are
not gated out, an intermediate edit may still receive positive local credit
when its after-state improves downstream answer success, or negative credit
when it damages it. If every paired downstream answer receives the same task
reward, the estimator correctly supplies zero local utility; increasing the
number or diversity of rollouts is then necessary rather than substituting a
structural heuristic.

The counterfactual prompt is a graph-conditioned downstream QA probe, not a
full replay of search/tool continuation from the edit point. That keeps the
counterfactual tractable and isolates the usefulness of the serialized memory
available to the answering policy.

## Diagnostic current-policy answer-likelihood utility

Set `GRAPH_RPO_CREDIT_BACKEND=old_policy_answer_likelihood` to compute graph
utility without a separate learned evaluator. For every successful episode,
the PPO driver teacher-forces the benchmark's known correct answer once with
the graph immediately before an edit and once with the graph immediately after
it. The utility increment is the length-normalized answer log-likelihood
difference, less any configured operation cost, clipped by
`GRAPH_RPO_DELTA_MAX`.

The scoring pass uses the current actor under `no_grad` before `update_actor`
and caches ordinary scalar values. Thus one rollout batch is scored by a fixed
old-policy snapshot even though the actor changes between GRPO steps. The
generated answer is never used as the scoring target. This backend is retained
as a cheaper diagnostic ablation rather than the formal graph-value estimator,
so start with a small coefficient such as `GRAPH_RPO_ALPHA=0.1` and
`GRAPH_RPO_DELTA_MAX=0.25`, retain the verified terminal-outcome gate, and
compare against `GRAPH_RPO_ALPHA=0` in the formal ablation.

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

For a quick target-domain bootstrap from the completed 20-step BrowseComp
rollout pilot, run
`scripts/pilot_train_graph_evaluator_from_bc_rollouts_4node_idev.sh` inside a
four-node allocation. It trains a small independent classifier on one node,
logs the evaluator run to W&B, calibrates it on a question-disjoint validation
partition, requires at least five validation questions with both successful and
failed trajectories, and measures episode-balanced ranking within each such
question. The policy-use gate requires both global AUROC discrimination, a
Brier score no worse than the constant-prevalence baseline, and non-random
within-question macro AUROC. This is only the evaluator bootstrap stage for
alternating training; it does not update the actor or use `bc_test.parquet`.

Build the target-domain evaluator on a four- or five-node Vista allocation with
`scripts/build_bc_graph_rpo_evaluator_qwen3_8b_4node.sh`. The workflow runs the
original Qwen3-8B policy in validation-only mode on `bc_train.parquet`, samples
multiple episodes per question, audits the fixed BrowseComp judge labels, and
then fine-tunes the pilot checkpoint. It never updates the target policy and
rejects `bc_test.parquet`. Derived target rollouts, evaluator Parquet data, and
the frozen model are stored in separate `$SCRATCH` roots. The final directory
contains both `graph_rpo_calibration.json` and `graph_rpo_evaluation.json`;
inspect the question-disjoint validation discrimination and calibration metrics
before fixing that exact checkpoint path for the policy run.

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
`scripts/smoke_train_bc_ctxgraph_8b_graphrpo_5node_idev.sh` performs one
optimizer step with the selected credit backend. It defaults to paired
old-policy counterfactual QA.

On an existing four-node allocation, run the original Qwen3-8B zero-shot
judge-audit variant, including rollout persistence and post-run audit, with:

```bash
bash scripts/smoke_train_bc_ctxgraph_8b_graphrpo_qwen3_8b_4node_judge_audit.sh
```

Run the formal paired-counterfactual backend for two optimizer steps with:

```bash
bash scripts/smoke_train_bc_ctxgraph_8b_graphrpo_counterfactual_4node_2step.sh
```

To test the older teacher-forced current-policy likelihood ablation for two
optimizer steps from the original Qwen3-8B snapshot, with required W&B logging
and post-run signal checks, use:

```bash
bash scripts/smoke_train_bc_ctxgraph_8b_graphrpo_old_policy_4node_2step.sh
```
