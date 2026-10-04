# WideSearch and ScienceWorld with Qwen3.5-9B

Both benchmarks run through the existing FoldAgent and isolated ContextGraph
executors. The new launcher owns two model servers and two evaluator processes,
records logs for direct `idev` runs as well as `sbatch`, and saves each task before
starting the next one. Neither benchmark needs Docker.

This is a new evaluation integration, not a published performance result.
Local checks exercised both agent loops with deterministic model responses,
the real ScienceWorld JVM, and the pinned WideSearch evaluator with a fake judge.
Vista GPU execution, paid live search, and paid judge calls remain unverified.

## Protocol

| Setting | ScienceWorld | WideSearch |
| --- | --- | --- |
| Data | ScienceWorld 1.2.3 official test variations | ByteDance-Seed/WideSearch full split |
| Full size | 1,819 variations across 30 task types, enumerated from the installed simulator | 200 queries (100 English, 100 Chinese) |
| Environment | Python + Java simulator; no simplifications, no gold action paths | Tavily live search and page extraction; no BC-P corpus |
| Reported metrics | Mean final score (0–100), raw score per episode, count of score-100 successes | Official whole-table accuracy (0/1), row and item precision/recall/F1 |
| Judge | None | `gpt-4.1-2025-04-14`, temperature 0, 10,240 completion tokens |
| Default environment cap | 100 executed actions, including free look/inventory calls; simulator moves recorded separately | 100 main turns; page text limited to 20,000 characters, search snippets to 2,000 each |

Model: `Qwen/Qwen3.5-9B`, snapshot
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`. Default context is 65,536 tokens,
with an 8,192-token prompt allowance and 57,344-token main response budget.
Decoding is temperature 0, top-p 1, thinking enabled; task seeds are stable and
identical across methods. Both methods have the same branch settings (up to ten
sessions, branch budget 57,344). These are **main/branch budgets, not a pooled
total-token cap**; compare request logs as well as accuracy for cost comparisons.
WideSearch reserves 4,096 tokens for final submission. ContextGraph uses the
existing balanced structured controller; FoldAgent uses its existing branch loop.
No RL shaping reward is reported as a benchmark score.

ScienceWorld branches act sequentially in the same episode; there is no branch
rollback, simulator cloning, or independent parallel environment. Scores below
zero indicate simulator failure; aggregate scores clip these to zero while
preserving `raw_score`. `done` alone is **not** success. The adapter also now
returns the termination signal expected by the agent loop, preventing more model
calls after the episode ends. This corrects historical adapter semantics; old
ScienceWorld results should not be pooled with new results without re-evaluation.

WideSearch dataset revision is `6531a7e5b497d44c8912407e0cb3dc95bd98cc09`;
official evaluator revision is `9825ba7b140b71d81b364793f86dabe4cfed6749`.
The upstream parser, matching rules and aggregation are imported unchanged.
Only its cloud transport is replaced with an OpenAI-compatible client; malformed
judge responses and swallowed evaluator exceptions become infrastructure errors.
This is **official metrics with a custom agent/search provider**, not a claim of
reproducing the authors' complete agent protocol. Live web results can change:
each search/extract response is timestamped and saved, but it is not a frozen web
snapshot shared between methods.

Sources: [ScienceWorld](https://github.com/allenai/ScienceWorld),
[WideSearch code](https://github.com/ByteDance-Seed/WideSearch),
[WideSearch data](https://huggingface.co/datasets/ByteDance-Seed/WideSearch),
[Tavily search](https://docs.tavily.com/documentation/api-reference/endpoint/search).

## Vista: smoke first

Use the existing model checkpoint and working agent environment. The launcher
creates a private dependency overlay under `$SCRATCH/context-graph-agent-benchmarks/envs/`;
it does not install into `cxtgraph` or the existing SWE environment. Set
`BENCH_BASE_PYTHON` if your working agent Python differs from the default
`/scratch/09281/chc_1996/context-graph-swe/envs/agent-direct-Rs4ngP/bin/python`.
The separate model server retains the repository's established serving setup.
Initial setup requires access to PyPI/Hugging Face/GitHub; WideSearch also requires
outbound HTTPS from the evaluator nodes.

For ScienceWorld, Java must be available on all evaluator nodes through `PATH`
or `JAVA_HOME` (Java 17 was used for the local smoke). Check `java -version`.
For WideSearch, export `TAVILY_API_KEY` and `OPENAI_API_KEY` before submission.
Search and judge calls may incur charges. An optional OpenAI-compatible judge
endpoint is `WIDESEARCH_JUDGE_BASE_URL`; it is recorded in the manifest.
Each evaluator makes one small search and judge preflight call before generation
to catch authentication/network/model errors before spending the full run budget.

From a Vista login node (Bash):

```bash
cd /work/09281/chc_1996/vista/context-graph
git pull --ff-only origin master
mkdir -p logs
BENCH_SAMPLES=2 sbatch scripts/eval_agent_benchmarks_qwen35_9b_4node.sbatch scienceworld both
BENCH_SAMPLES=2 sbatch scripts/eval_agent_benchmarks_qwen35_9b_4node.sbatch widesearch both
```

These are two separate four-node jobs. `both` runs the same two tasks for each
method, one server/evaluator pair per method. `BENCH_SAMPLES` selects the first N
sorted task IDs and is intended for a plumbing smoke, not a representative subset.
Do not update the shared checkout while a job is running.

Inside an existing four-node `idev` allocation, use `bash` instead of `sbatch`:

```bash
BENCH_SAMPLES=2 bash scripts/eval_agent_benchmarks_qwen35_9b_4node.sbatch scienceworld both
```

After the smoke, use `BENCH_SAMPLES=-1` for all tasks. If you have two separate
four-node allocations, each method can use two shards. For example:

```bash
BENCH_SAMPLES=-1 sbatch scripts/eval_agent_benchmarks_qwen35_9b_4node.sbatch widesearch contextgraph
BENCH_SAMPLES=-1 sbatch scripts/eval_agent_benchmarks_qwen35_9b_4node.sbatch widesearch foldagent
```

Replace `widesearch` with `scienceworld` for its full test split. Full ScienceWorld
has substantially more tasks; the launcher does not promise completion within a
single allocation. `BENCH_CONTEXT_LENGTH=32768` is also supported. Keep context,
`BENCH_MAX_STEPS`, checkpoint and judge settings identical when comparing methods.

## Results, trajectories and resume

The start banner prints the exact run path:

```text
$SCRATCH/context-graph-agent-benchmarks/runs/<benchmark>-9b-<mode>-<job>-<suffix>/
  suite.log
  environment.txt
  server-<label>.log
  evaluator-<label>.log
  <label>/manifest.json
  <label>/summary.json
  <label>/instances/<task-key>/result.json
  <label>/instances/<task-key>/attempt-*/generation.log
  <label>/instances/<task-key>/attempt-*/tools.jsonl
  <label>/instances/<task-key>/attempt-*/requests.jsonl
  <label>/instances/<task-key>/attempt-*/trajectory.json
```

Labels are `contextgraph` and `foldagent` for `both`, or `contextgraph-0/1` /
`foldagent-0/1` for a sharded single-method run. `trajectory.json` contains the
exported main transcript and graph diagnostics; `requests.jsonl` records token
IDs for model calls, including branches/controllers, and `tools.jsonl` records
environment interactions. A killed task may have only these incremental logs.
WideSearch also saves `prediction.md`, `grading.log`, `judge.jsonl`, and the
official per-row `official-details.csv` when available. Gold references are kept
in the prepared data directory and never included in the agent's task input.

Set `BENCH_RUN_DIR` to the path printed by the launcher, then inspect:

```bash
cat "$BENCH_RUN_DIR"/*/summary.json
tail -n 30 -f "$BENCH_RUN_DIR"/evaluator-*.log
```

`mean_score_graded` describes only successfully graded tasks. `mean_score` remains
null until all selected tasks are graded; infrastructure failures are counted
separately, never silently turned into model failures. When combining shards,
weight means by `graded`, not by shard count. Whole-table accuracy is strict, so
also inspect WideSearch F1 metrics.

To resume a stopped **single-method WideSearch run**, set `BENCH_RUN_DIR` to its
existing directory and keep the original method/sample selection/configuration:

```bash
BENCH_RUN_DIR=/scratch/09281/chc_1996/context-graph-agent-benchmarks/runs/EXACT_EXISTING_RUN BENCH_SAMPLES=-1 sbatch scripts/eval_agent_benchmarks_qwen35_9b_4node.sbatch widesearch contextgraph
```

Completed records are skipped. An interrupted task without a result restarts in a
new attempt directory. Set `BENCH_RETRY_ERRORS=1` to retry infrastructure errors;
previous attempts remain on disk. Never retry already graded wrong answers when
reporting a single-attempt score. Resume rejects changes to the recorded code,
data, model, dependencies or budgets.
If only WideSearch grading failed, its generated prediction and trajectory are
reused for the grading retry rather than generating a new answer.

## Local or custom deployment

`scripts/prepare_agent_benchmarks.py <benchmark> --output <directory>` creates
`tasks.json`; preparation deliberately refuses to overwrite an existing bundle.
`scripts/eval_agent_benchmarks.py` accepts `--data`, `--output`, `--endpoint`,
`--model-path`, `--method`, `--samples`, and shard arguments. The endpoint must be
the pinned Qwen3.5-9B server with token-ID completions; this is not a generic
text-only Chat Completions evaluator. Use `--help` for limits and retry options.
