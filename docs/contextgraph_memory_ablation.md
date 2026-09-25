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

Local pure-Python tests do not replace Vista runtime/import validation. A skipped
replay test means executor equivalence has not yet been verified in that runtime.
The suite fails its dependency preflight instead of accepting this skip.
