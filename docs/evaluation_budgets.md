# Working-context evaluation protocol

The evaluator defaults to `AGENT_BUDGET_MODE=working_context`. This applies to
BC-P (ReAct, FoldAgent, ContextGraph, AgentFold, SUPO, MemoBrain, A-MEM), SWE,
the six methods exposed by the ScienceWorld/DiscoveryWorld launcher, and the
explicit BC-P/ScienceWorld SFT collector. Training configs without the new
setting keep their previous behavior; this change does not redefine RL budgets.

## What a 32K or 64K limit means

Each actor, branch, memory-helper and finalization request must fit its fully
rendered input **plus generated output** into the configured window. Repeated
requests are not added together to decide whether that window is full. When a
method compresses or reconstructs its working context, it can use the recovered
space. Main and branch requests use the same physical window size. A branch's
past generation does not consume the parent's free context space.

The run still has turn/request and time limits. Memory helpers count as requests;
this is not a grant of unlimited helper work. SUPO's configured summary-count
limit, memory retrieval sizes, output caps, and graph decision ownership are
unchanged. Explicit graph compaction remains a model-selected action. This patch
does not force the teacher to use graph tools or relax SFT acceptance gates.

In window mode, `graph_compact` checks its complete active-graph snapshot against
the physical window, reserving the generation prefix and at least 129 tokens of
headroom. The initial `prompt_length` allocation (often 8K) is not a separate
snapshot cap. The model-visible fit estimate and executor use the same bound;
the new segment retains the complete snapshot in both inference and export.
Oversized snapshots are rejected without dropping nodes. Legacy cumulative mode
retains its prompt cap. A merge changes the graph; only a model-requested
`graph_compact` replaces conversation history.

## Finalization and reporting

* BC-P enables emergency finalization for all seven methods.
* SWE disables emergency finalization for all methods: the harness grades the
  actual patch after a natural finish, window limit, turn limit, or timeout.
* ScienceWorld/DiscoveryWorld terminate through the environment or budget rules;
  no synthetic finish action is added.

SUPO `zero_shot_rollout_adaptation_v5` treats `supo_max_summaries` as a cap on
summary calls, not an episode termination condition. When the threshold is
crossed after that cap, the latest executed action/observation is retained using
normal budget-aware observation fitting. With a configured final-answer reserve
(BC-P), the next counted request enters the existing finalizer. Without one
(default SWE and both simulators), task actions continue without more summaries
until normal environment/window/turn/time termination. No actions are replayed,
no additional summaries are granted, and no simulator `finish` is invented.

`hit_summary_limit` now records that the cap was encountered, even if the task
subsequently finishes; it is no longer a termination reason. Finalization audit
records identify `final_trigger=summary_limit`. Earlier v4 runs that terminated
with `summary_limit` require rerunning for comparable results; this code change
does not revise their saved scores or establish an accuracy improvement.

Results record `token_budget_mode`, `working_context_limit`,
`episode_token_limit` (null in window mode), `finalizer_enabled`, and
`finalizer_attempted`. `forced_finish` records a successful forced submission;
it is not interchangeable with an attempt. In tables, report forced-finalization
rate only where enabled. Disabled or unrecorded finalization should be labelled
N/A, not interpreted as 100% natural completion.

`total_model_tokens` sums actual input and output IDs over all actor/branch/helper
requests, including repeated input. `peak_request_tokens` is the maximum input
plus output of one request; it is not the historical table's input-only Peak
column. The four memory baseline loops also record their original
`cumulative_accounted_tokens` ledger (generation + visible observations + control
instructions). Core agents expose `response_ledger_tokens` separately: old
controller projections can replace retained history, so this field must not be
treated as total computation. Do not infer equal FLOPs or equal wall time from
equal windows. Native environment action limits and helper request overhead
remain relevant experimental settings.

## Existing results and reruns

The previous four-baseline implementation capped cumulative work at
`val_response_length` (usually context minus 8,192). Some historical core-agent
paths allowed much more total branch work. Their numerical context labels did
not establish a matched protocol. Existing Table 2 numbers must be regenerated
under one protocol before drawing a matched-window comparison; this code change
does not validate or improve the old scores.

Run in a fresh output directory. Evaluator manifests save the resolved mode;
resume rejects changing it. Do not reuse an old `SWE_RUN_DIR`. Existing jobs and
saved results are not modified by this patch. Local regression tests validate
budget/dispatch behavior; a new Vista smoke and full evaluation are still needed.

On an existing four-node Vista allocation, test one method first:

```bash
git pull --ff-only
AGENT_BUDGET_MODE=working_context SAMPLES=2 bash scripts/smoke_stateful_memory_qwen35_9b_4node_idev.sh swe-lite amem
```

For BC-P, with the existing local corpus/judge setup:

```bash
AGENT_BUDGET_MODE=working_context SAMPLES=3 DATA_PATH="$PWD/data/bc_test.parquet" bash scripts/eval_bcp_qwen35_9b_4node_idev.sh amem
```

To reproduce an old budget convention explicitly, set
`AGENT_BUDGET_MODE=legacy_cumulative`. That is a reproduction switch, not a fair
matched-compute protocol across all historical methods. Keep those results apart
from window-mode runs.
