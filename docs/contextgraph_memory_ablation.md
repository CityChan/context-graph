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
