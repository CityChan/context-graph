# ScienceWorld with Qwen3.5-9B

The four-node launcher prepares ScienceWorld, starts two inference servers,
runs ContextGraph and FoldAgent, and saves per-task scores and trajectories.
ScienceWorld uses a local Java simulator; Docker and external search or judge
API keys are not required.

## Protocol and validation

| Setting | Value |
| --- | --- |
| Simulator | ScienceWorld 1.2.3, no simplifications or gold action paths |
| Data | Official test split: 1,819 variations across 30 task types |
| Model | Qwen/Qwen3.5-9B, snapshot `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| Context | 65,536 tokens by default; 32,768 also supported |
| Decoding | Temperature 0, top-p 1, thinking enabled, stable per-task seeds |
| Environment cap | 100 executed actions, including look/inventory; simulator moves recorded separately |
| Scores | Mean environment score on 0–100 scale; success requires score 100 |

At 64K, both methods use an 8,192-token prompt allowance and a 57,344-token
main response budget. Branch settings match: up to ten sessions and a
57,344-token branch budget. These are main/branch budgets, not a pooled total
token cap. Use request logs when comparing inference costs. ContextGraph uses
the balanced structured controller; FoldAgent uses its existing branch loop.
RL shaping rewards are not benchmark scores.

New server launches default to `SERVER_ENFORCE_EAGER=1` (`--enforce-eager`).
Explicit `SERVER_ENFORCE_EAGER=0` opts out and clears `TORCHDYNAMO_DISABLE` in
the server process. Both methods must use the same setting. The requested
setting is recorded in the evaluator manifest; it does not attest the mode of
an independently started endpoint. Changing it requires a new run directory.
Existing jobs and checkouts pinned to older commits are not updated.

After each task, the evaluator scans that attempt's `requests.jsonl` output IDs.
A request containing at least 20 consecutive token IDs of zero is flagged once.
`summary.json` records `model_requests`, `degenerate_requests`, and
`degenerate_request_rate`. Strictly more than 1% at any progress checkpoint
saves the summary with `generation_quality_passed=false` and raises an error,
which makes the batch exit nonzero. Exactly 1% is allowed. These are request
counts, not token counts, and only current attempts contribute. Missing audit
evidence is reported as `generation_quality_passed=null` with
`generation_unaudited_records`; it is not evidence of clean generation.
This heuristic detects the observed output pattern, not its hardware cause.

### Read-only paired diagnosis

`scripts/audit_scienceworld_pair.py` compares only tasks graded by both methods,
checks manifests, breaks down scores by task type, and reports termination,
environment steps, controller calls and elapsed time. It extracts bounded tool
and message tails for the largest score gaps in both directions. With
`--scan-requests`, those selected cases also get raw token and controller-request
counts; this is not a full-run token-degeneration certification. The script uses
only the Python standard library and resolves saved Vista attempt paths relative
to each result, so copied Windows artifacts work without editing results.

```bash
python scripts/audit_scienceworld_pair.py --contextgraph /path/to/contextgraph-run --foldagent /path/to/foldagent-run --output /path/outside/runs/scienceworld-audit.json --scan-requests
```

The output must be new and outside both runs. Completed result snapshots are
read without mutating running jobs. Missing metrics remain unknown; protocol
mismatches are explicitly reported and must be resolved before treating the
comparison as matched.

New ScienceWorld runs set `plugin.graph_controller_counts_as_turn=false` for
both methods and record it in the manifest. ContextGraph controller checkpoints
(including pass or rejected controller responses) no longer consume the
100-turn task budget. They still consume the main token/context budget and
session time. Ordinary main-agent responses, summaries and branch calls retain
their existing turn accounting; an out-of-mode graph request in the ordinary
task channel is still a counted, rejected response. The simulator retains its
independent 100-action cap. This gives more task turns, not unlimited compute
or a guarantee of 100 executed actions.

Per-task stats expose `turn_budget_used`, `graph_controller_turns`, and
`graph_controller_counts_as_turn`; request logs retain every controller call.
The generic graph agent defaults the flag to true for compatibility outside
ScienceWorld. Existing pinned jobs retain their protocol. Use a fresh output
directory for new runs; do not mix these results with the historical protocol.

Historically there were two distinct limits: 100 simulator actions and 100
agent-loop model turns, with consolidation calls incrementing the latter. A controlled
test using both real agent loops, a fixed action policy, no branches and an
enlarged token budget (the test tokenizer counts characters) demonstrates that
16 controller calls can leave only 84 simulator actions. A synthetic task that
requires 90 actions succeeds with FoldAgent and with ContextGraph checkpoints
disabled, but fails under the historical five-turn checkpoint schedule.
With controller turns excluded, the same test succeeds at action 90 with
17 controller calls; a separate boundary case reaches action 100 with 19
controller calls and still stops at the simulator cap.
This establishes a mechanism, not its contribution to any production score
gap. Inspect real `hit_max_turn`, `environment_steps`, and `consol_attempts` on
common tasks when interpreting results. An equal model-call budget and an equal
environment-action opportunity are different comparison protocols.

Branches act sequentially in the same episode. There is no simulator cloning
or rollback. Negative terminal scores are clipped to zero for aggregation and
preserved as `raw_score`. A terminal `done` flag alone does not indicate success.
The adapter signals termination to the agent loop to stop further model calls.

Local verification covers the real ScienceWorld JVM and both agent loops with
deterministic model responses. This is not evidence of a completed Vista GPU
evaluation or model performance. Source: [ScienceWorld](https://github.com/allenai/ScienceWorld).

## Vista setup and smoke test

Java must be available on evaluator nodes through `PATH` or `JAVA_HOME`;
Java 17 was used for local verification. Check `java -version` first.

The launcher creates a private dependency overlay at
`$SCRATCH/context-graph-agent-benchmarks/envs/scienceworld-api`, leaving
`cxtgraph` and the existing SWE environment intact. Initial setup needs PyPI
access. The default parent Python is
`/scratch/09281/chc_1996/context-graph-swe/envs/agent-direct-Rs4ngP/bin/python`;
override `BENCH_BASE_PYTHON` if needed. `BENCH_AGENT_ENV` selects another overlay.
The model server uses the repository's established serving environment.

From a Vista login node (Bash):

```bash
cd /work/09281/chc_1996/vista/context-graph
git pull --ff-only origin master
mkdir -p logs
BENCH_SAMPLES=2 sbatch scripts/eval_agent_benchmarks_qwen35_9b_4node.sbatch scienceworld both
```

Inside an existing four-node `idev` allocation:

```bash
env -u BENCH_RUN_DIR -u BENCH_DATA BENCH_SAMPLES=2 bash scripts/eval_agent_benchmarks_qwen35_9b_4node.sbatch scienceworld both
```

`both` runs the same selected tasks for each method, with one server/evaluator
pair per method. `BENCH_SAMPLES` selects the first N sorted task IDs for a smoke
test, not a representative subset. Avoid updating the shared checkout during a run.

With two separate four-node allocations, run one method in each allocation:

```bash
env -u BENCH_RUN_DIR -u BENCH_DATA BENCH_SAMPLES=2 bash scripts/eval_agent_benchmarks_qwen35_9b_4node.sbatch scienceworld contextgraph
```

```bash
env -u BENCH_RUN_DIR -u BENCH_DATA BENCH_SAMPLES=2 bash scripts/eval_agent_benchmarks_qwen35_9b_4node.sbatch scienceworld foldagent
```

Each single-method run splits tasks into two shards. After smoke tests pass,
set `BENCH_SAMPLES=-1` for the full test split; completion within one allocation
is not guaranteed. Keep `BENCH_CONTEXT_LENGTH`, `BENCH_MAX_STEPS`, model,
dependencies and code identical between methods.

## Results and resume

The startup banner prints the exact artifact directory:

```text
$SCRATCH/context-graph-agent-benchmarks/runs/scienceworld-9b-<mode>-<job>-<suffix>/
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

Labels are `contextgraph` and `foldagent` for `both`, or `contextgraph-0/1`
and `foldagent-0/1` for single-method runs. `trajectory.json` saves the main
transcript and graph diagnostics; `requests.jsonl` records token IDs including
branch/controller calls; `tools.jsonl` records environment interactions.
Interrupted tasks may have only incremental logs.

Set `BENCH_RUN_DIR` to the printed run directory, then inspect:

```bash
cat "$BENCH_RUN_DIR"/*/summary.json
tail -n 30 -f "$BENCH_RUN_DIR"/evaluator-*.log
```

`mean_score_graded` covers graded tasks only. `mean_score` stays null until all
selected tasks are graded. Infrastructure failures are reported separately.
Combine shard means weighted by `graded`; sum `successes` to count solved tasks.

Resume a stopped run using its exact directory and original method/sample count:

```bash
BENCH_RUN_DIR=/scratch/09281/chc_1996/context-graph-agent-benchmarks/runs/EXACT_EXISTING_RUN BENCH_SAMPLES=-1 sbatch scripts/eval_agent_benchmarks_qwen35_9b_4node.sbatch scienceworld contextgraph
```

Completed records are skipped. Interrupted tasks start a new attempt directory.
Set `BENCH_RETRY_ERRORS=1` to retry infrastructure errors while retaining previous
attempts. Resume rejects changes to code, data, model, dependencies or budgets;
use the original revision to resume an older run. Do not retry graded incorrect
answers when reporting single-attempt scores.

## Custom deployment

`scripts/prepare_agent_benchmarks.py scienceworld --output <directory>` prepares
`tasks.json` and refuses to overwrite an existing bundle.
`scripts/eval_agent_benchmarks.py scienceworld --help` lists endpoint, model,
data, output, method, shard, budget and retry options. The evaluator requires
the pinned Qwen3.5-9B checkpoint with token-ID completions.
