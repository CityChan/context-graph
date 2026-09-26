# ContextGraph memory diagnosis and paired evaluation

The opt-in `contextgraph_memory_mode` separates the historical implementation
(`legacy`, still the default) from `repaired` and the `foldagent` equivalence
control. These are implementation diagnostics, not established performance gains.

`foldagent` delegates to the original FoldAgent executor before constructing a
graph or environment. It copies request metadata/config, changes `search_graph`
to `search_branch`, restores FoldAgent process labels, and disables graph tool
format repair. It supports search evaluation only. Runtime tests replay a fixed
branch/search/return/finish tape and compare actual requests and output tokens.

`repaired` changes the following together (a bundled repair, not an isolated
ablation of each mechanism):

- Preserve observation history until 75% of the prompt-plus-response budget is
  occupied. Keep original branch return summaries without the 2000-character cut.
- Do not inject the graph inventory after every environment action by default.
  Controller acknowledgments still carry the graph state.
- Allow pass at every checkpoint and permit select under the balanced policy.
- Prioritize explicit focus and causal/semantic neighbors during retrieval;
  absent focus, expand one lexical query seed. Temporal/root adjacency does not
  make all observations equally important. These priorities are heuristics.
- Select a query-relevant, overlapping source passage when a long source does
  not fit. Enforce the original token budget using the model tokenizer.
- Down-rank pruned evidence while retaining archival recoverability; prune the
  oldest equal-value observation first. The reversed legacy order remains only
  in the legacy control for reproduction.

Run on Vista (self-submits from a login node, or uses an existing four-node idev):

```bash
bash scripts/eval_bcp_memory_ablation_4node.sh
```

The suite first requires the real executor replay test to pass. It uniformly
samples 24 BC-P questions with seed 42, without using outcomes, writes the source
hash, original row indices and Git commit, and runs FoldAgent, equivalent mode,
legacy ContextGraph and repaired ContextGraph sequentially. Each variant gets
the same rows, base Qwen3-8B, 64K budget, greedy decoding, 100 turns, 10 sessions,
1024-token finalizer reserve and no StructMem. Only workflow metadata differs.
Set `SAMPLES=-1` for the full 150-question follow-up after inspecting diagnostics.

With two separate four-node idev allocations, set `ABLATION_GROUP=controls`
in one to run FoldAgent and equivalent, and `ABLATION_GROUP=graph` in the other
to run legacy and repaired. Each allocation runs its two variants sequentially;
the allocations can run concurrently. Use distinct, new `RUN_ROOT` directories.
Both runs must use the same clean Git commit, source data, `SAMPLES` and `SEED`,
and evaluation configuration. Each independently prepares the same deterministic
subset. Group runs defer the four-way audit until both finish. The default
`ABLATION_GROUP=all` retains the original sequential four-variant workflow.

For example, after updating the checkout once, run in the first idev:

```bash
ABLATION_GROUP=controls RUN_ROOT="$PWD/outputs/memory-pair-20260926-controls" bash scripts/eval_bcp_memory_ablation_4node.sh
```

Run in the second idev from the same checkout:

```bash
ABLATION_GROUP=graph RUN_ROOT="$PWD/outputs/memory-pair-20260926-graph" bash scripts/eval_bcp_memory_ablation_4node.sh
```

When both finish, in the `cxtgraph` environment run:

```bash
python scripts/audit_contextgraph_ablation.py outputs/memory-pair-20260926-controls --peer-root outputs/memory-pair-20260926-graph
```

The joint audit rejects mismatched commit, source hash, sample indices, seed,
sample count or selection method, and missing or duplicate variants. It reads
the original artifacts in place and writes reports only to the first directory.
These checks establish subset/provenance consistency, not equality of every
runtime or server-side setting. Existing output directories are rejected to
avoid mixing repeated runs; choose a new pair of directory names for a rerun.

Outputs live in a new `outputs/memory-ablation-<job>-<timestamp>/` directory.
Each validation JSONL includes task IDs, main and branch model request snapshots
(messages, exact input IDs, completion constraints/budgets), final messages,
terminal task reward and graph traces where available. Capturing full requests
is intentionally expensive and disabled for ordinary runs. It is more reliable
than the final rewritten chat for diagnosing memory visibility.

`audit.json` checks coverage, counts main-episode task success, and lists the
first differing FoldAgent/equivalent main request and any branch divergence.
Separate GPU runs can diverge despite greedy decoding; inspect preceding model
outputs before attributing later request differences to the executor.

To diagnose an existing run without GPUs or new generations, run:

```bash
python scripts/audit_contextgraph_ablation.py outputs/memory-ablation-1020286-20260925_113412
```

The audit also writes `equivalence_diagnostics.json` with per-task main request
differences, branch key differences, and comparisons of shared branch requests.
Request indices are zero-based: index 1 means the second request, with the first
request matching. The compact stdout summary counts diagnostic categories.
Details include the first changed message and content window, token mismatch
offset and window, and changed request budgets or completion arguments.

An appended assistant-message difference is an observed history difference, not
proof of nondeterministic raw generation: postprocessing can also change it.
An observation difference calls for tool/environment and orchestration review.
Identical messages with different tokens or budgets call for prompt assembly or
configuration review. Rewritten histories are classified separately because
adjacent snapshots do not necessarily retain the intervening raw output.
Matching snapshots do not verify all server-side sampling settings. Branches
are matched only by their stored keys; renamed branches are not assumed equal.
The evaluation commit remains the original manifest commit. This command
refreshes the two audit reports only and retains the existing JSONL evidence.
Exit status 1 still means request equivalence failed, even when diagnostics were
written successfully; exit status 0 means the recorded requests matched.

Local pure-Python tests do not replace Vista runtime/import validation. A skipped
replay test means executor equivalence has not yet been verified in that runtime.
The suite fails its dependency preflight instead of accepting this skip.

### Qwen3 observation-token repair

The September 25 run `memory-ablation-1020286-20260925_113412` exposed a
shared context-construction bug: task 129's branch summary was present in
`messages` but absent from the next recorded `input_ids` in both FoldAgent and
equivalent mode. Qwen3's template rewrote historical thinking when a new user
message arrived. The rendered prefix shrank from 5731 to 5374 tokens, so slicing
the new render at the old prefix length produced an empty observation.

Incremental construction now checks prefix equality. On a history rewrite it
renders the new turn independently and requires that it exactly match a suffix
of the full rendered conversation. Unsupported context-dependent templates
raise an error instead of silently losing evidence. Existing generated tokens,
log probabilities and policy masks remain unchanged; this preserves the stored
rollout trajectory rather than replacing it with a fresh full-chat rendering.

Run the cached, real-Qwen3-tokenizer regression check without GPUs or generation:

```bash
bash scripts/check_qwen3_observation_tokens.sh
```

The suite now requires this check before evaluation. Tests cover short branch
returns, long search observations, replacement/rollback, and training alignment.
Retain the original run as diagnostic evidence; the repair does not update old
JSONL inputs or scores. Rerun matched variants under a new output directory before
assessing performance. Initial generation divergence is a separate open issue.
