# DiscoveryBench: matched Qwen3.5-9B evaluation

Run ContextGraph and FoldAgent in one four-node Vista allocation. Nodes 0/1
serve/evaluate ContextGraph; nodes 2/3 serve/evaluate FoldAgent. Each method
evaluates all 239 real-test questions, not one half of the dataset. No training
or retrieval server is involved. The local Python execution environment is the
existing DiscoveryBench sandbox, not the SWE-bench Apptainer environment.

From a Vista login node:

```bash
cd /work/09281/chc_1996/vista/context-graph && git pull --ff-only origin master && env -u DB_RUN_DIR bash scripts/submit_eval_discoverybench_qwen35_9b.sh
```

The request is four GH nodes for 48 hours; completion within the allocation is
not guaranteed. To validate first, prefix the submission with `DB_SAMPLES=2`.
The runner requires all 239 unique real-test rows before selecting smoke samples.

## Two existing four-node idev allocations

Run one command in each allocation's compute-node shell. The launcher starts
two model servers and two evaluators per allocation automatically. Each method
covers all 239 real-test questions, split into disjoint shards of 120 and 119.
Both methods use the same sorted task list and task-derived seeds.

ContextGraph allocation:

```bash
cd /work/09281/chc_1996/vista/context-graph && git pull --ff-only origin master && env -u DB_RUN_DIR DB_CONTEXT_LENGTH=65536 DB_SAMPLES=-1 bash scripts/eval_discoverybench_qwen35_9b_4node.sbatch contextgraph
```

FoldAgent allocation:

```bash
cd /work/09281/chc_1996/vista/context-graph && git pull --ff-only origin master && env -u DB_RUN_DIR DB_CONTEXT_LENGTH=65536 DB_SAMPLES=-1 bash scripts/eval_discoverybench_qwen35_9b_4node.sbatch foldagent
```

These commands use the existing allocations; `bash` ignores the `#SBATCH`
48-hour request, so the original idev time limits still apply. Keep both runs
on the same checkout commit. `DB_SAMPLES` limits the global task list before
sharding; use 2 to smoke-test one task per evaluator.

The printed run directory contains `suite.log`,
`evaluator-contextgraph-{0,1}.log`, `server-contextgraph-{0,1}.log`, and
`contextgraph-{0,1}/summary.json` (replace `contextgraph` with `foldagent`
for that allocation). Each shard has the same per-task artifact layout below.
Add completed/graded counts across shards; combine HMS using graded task counts
as weights, not an unweighted average. The method-wide final HMS requires both
shards to be fully graded. For resume, retain the same method and run directory;
the manifest checks shard identity and the global selection as well as protocol.

The submit helper also accepts `contextgraph` or `foldagent` for a separate
four-node batch job. Omitting the method retains the original two-method layout.

The launcher reuses the cached Qwen3.5-9B revision
`c202236235762e1c871ad0ccb60c8ee5ba337b9a` and `deepseek_v4` server runtime used
by the SWE evaluations. It creates a private scientific-Python overlay under
`$SCRATCH/context-graph-discovery/envs/qwen35-api`, using the known-working SWE
agent Python as its parent package path. Existing environments are not modified.
`DB_AGENT_ENV` and `DB_BASE_PYTHON` override those locations. Dependencies are
installed on first use and package versions are recorded. Package installation
and initial dataset download require network access. `DB_DATA` defaults to
`data/discoverybench_real_test_code.parquet`; if absent the existing loader
downloads/prepares real-test once. Both methods use that same file. Source
parquet and input-file hashes are recorded, rather than assuming a mutable
upstream snapshot is identical across runs.

HMS scoring needs the existing OpenAI or Azure judge credentials. The launcher
loads `.openai_env` using the established Vista locations when credentials are
not already exported; it never prints their values. OpenAI defaults to
`DISCOVERYBENCH_JUDGE_MODEL=gpt-5-nano`; Azure uses its configured deployment.
A real judge request, package imports and dataset checks precede server startup.
Judge identity is separate from the local Qwen generation endpoint.

## Matched protocol and scoring

- 65,536 context tokens by default (`DB_CONTEXT_LENGTH=32768` also supported).
  Prompt allowance 8,192; response/history budget 57,344 at 64K.
- Maximum 100 turns; 2,048 generated tokens per ordinary call; summary restart
  disabled; deterministic task-derived seeds based on seed 42.
- Existing `code_graph` isolated ContextGraph with balanced structured controller,
  and `code_branch` FoldAgent. No training-time reward shaping is scored.
- Each task runs in a fresh process to isolate the Python sandbox's cwd and
  execution state. Input data is staged in its work directory; all branch
  trajectories and token-ID request records are saved.
- The loop's internal format-only reward is discarded. After generation, the
  saved hypothesis/workflow is scored with the existing real HMS implementation.
  Missing or invalid prediction JSON receives zero; judge/process failures are
  recorded as infrastructure errors. HMS is a continuous score, not SWE's
  binary resolved count.
- `mean_hms` remains null until every selected task is graded. The partial
  `mean_hms_graded` is diagnostic only. Infrastructure errors cause job exit 2.

This is a new 9B/64K protocol, not a drop-in replication of the older
8B/12-turn scripts. Equal working-context limits do not imply equal total
inference cost: branches and graph-controller calls must also be counted from
`requests.jsonl` in comparisons. The existing Python sandbox is intended for
trusted benchmark execution; it is not a security boundary for hostile code.

## Logs and resume

Slurm logs: `logs/discoverybench-9b.JOBID.out` and `.err`. The startup log prints
the run directory under `$SCRATCH/context-graph-discovery/runs/`. It contains:

```text
suite.log
environment.txt
server-contextgraph.log
server-foldagent.log
evaluator-contextgraph.log
evaluator-foldagent.log
contextgraph/manifest.json
contextgraph/summary.json
contextgraph/instances/TASK_KEY/result.json
contextgraph/instances/TASK_KEY/attempt-*/generation.log
contextgraph/instances/TASK_KEY/attempt-*/requests.jsonl
contextgraph/instances/TASK_KEY/attempt-*/trajectory.json
contextgraph/instances/TASK_KEY/attempt-*/result.json
contextgraph/instances/TASK_KEY/attempt-*/workdir/pred_results/discovery_result.json
foldagent/... (same layout)
```

Task keys are portable sanitized names with a hash suffix; results retain the
original task ID. `trajectory.json` includes all returned branch trajectories.
`result.json` includes the actual HMS audit when scoring succeeds.

To resume after allocation expiry, keep the same checkout commit, environment,
data, model and budgets, and set `DB_RUN_DIR` to the printed paired run directory
when submitting. Do not pull a new commit first. Completed records, including
recorded infrastructure errors, are retained and skipped. An interrupted task
without a result gets a new attempt. Two launchers cannot own the same run.
The launcher stops only its owned Slurm steps at exit.

Local tests cover budget/method routing, task selection, scoring-error accounting,
branch export, resume and launch orchestration. Actual Qwen3.5-9B DiscoveryBench
execution on Vista is not yet verified; the earlier SWE run validates only the
reused model-serving path.
