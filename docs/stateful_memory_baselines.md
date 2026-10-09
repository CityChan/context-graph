# AgentFold and SUPO on SWE-bench Lite, DiscoveryWorld and ScienceWorld

Both are evaluation-only adaptations using Qwen3.5-9B, not the papers' trained
policies. SUPO's BC-P/GAIA behavior is unchanged; AgentFold's search v5 behavior
is also unchanged. Native AgentFold v6 adds constrained format retries for SWE,
DiscoveryWorld and ScienceWorld. It constrains suffix boundaries and native XML
tool structure, disables thinking only during recovery, and validates JSON and
tool arguments before dispatch. Python comparisons/indentation remain literal.
Rejected actions never execute, and the model still chooses and writes the fold.
Retries consume the existing turn/token budget and retain the three-error limit.
The supplied Vista logs verified SUPO mechanics on SWE and DiscoveryWorld; native
AgentFold v6 recovery and the focus_v3 prompt still require live GPU validation.

## Native environment contracts

AgentFold keeps the same model-selected contiguous suffix folding and strict
whole-block validation. Only the tool contract changes: `python_exec(code)` and
`finish(message)` for SWE, or `action(command)` for DiscoveryWorld and ScienceWorld. No local
search, branch or graph tools are added to these environments. Model-generated
folds never reset or fork the environment.

SWE uses the existing public-instance filter, sandbox, patch extraction and
grading harness. No gold patch/test enters model prompts. `finish` submits the
repository; its text is not a scored answer. Budget finalization disables thinking
and constrains a complete `finish(message)` call. Failures still clean up the
sandbox, and infrastructure errors never export a partial patch as valid.

DiscoveryWorld uses its existing official action/tick adapter, observation
profile and scorecards. There is no finish tool: task termination or the simulator
step limit ends the episode. Procedural scores and task success are retained;
knowledge evaluation is still not implemented. Private scorecards stay in grading
artifacts, not observations or memory prompts.

ScienceWorld uses its existing JVM adapter with `action(command)` containing a
single-line text command, not DiscoveryWorld JSON. No finish tool is exposed;
success, terminal task failure or the step limit terminates the episode. Raw negative
scores remain in task artifacts and are clipped to zero only by the existing
score aggregator. Simulator exceptions propagate as infrastructure errors. The
launcher explicitly uses `focus_v3`: focus selects a task target, not an inspection
action. It distinguishes `look at`/`look in` from target selection and adds a
short, labeled reminder after each nonterminal observation. Reminders enter the
visible token budget; raw simulator logs and terminal scores remain unchanged.
No action is blocked, corrected, retried or undone by this guidance. It uses no
gold targets or action lists. This profile differs from historical `legacy` and
`focus_v2` runs and must be reported for all compared methods. Both older profiles
remain available through the generic evaluator. Model adherence needs live validation.
Summary restarts preserve the latest executed action and observation, just as in
DiscoveryWorld. The simulator is never reset or replayed by memory maintenance.

## SUPO stateful overflow adaptation

BC-P continues to exclude a threshold-crossing round from working memory, as
before. In these stateful environments an executed edit/move cannot be forgotten
without losing the current state. The `retained_round_v1` profile instead summarizes
the preceding context and appends the actual crossing call and its shown observation
to the resumed context. Actions are never replayed or rolled back. Observations
are charged once and any budget truncation is explicit. If that retained round
plus summary cannot fit below the threshold, the adapter fails visibly; it never
silently drops current state. Provenance records this deviation from the paper.

All three environments use 64K context in the supplied launcher. SUPO defaults to
a 32K working-context threshold, two summaries and 1024 output tokens per summary.
The bounded summary format retains the existing v4 character cap. Summary requests
consume model turns, tokens and time but do not advance the simulator or execute
repository commands. Memory compression does not refund cumulative token budget.
DiscoveryWorld allows 200 model turns and at most 200 environment steps.
ScienceWorld allows 100 model turns and at most 100 environment steps. SWE uses
the existing runner's 100-turn default. These choices are recorded in task config.

## Four-node Vista smoke

Run one method in each allocation, or run them sequentially after the previous
launcher exits. Outputs are created under the checkout's `output/` directory;
inherited old run directories are ignored. The dedicated SWE environment and its
preparation/grading sub-environments must already exist; `SWE_AGENT_ENV` can select
the previously validated installation. DiscoveryWorld uses the existing private
Python 3.10 overlay selected from `cxtgraph`.

Launchers signal only their own jobs, then wait up to 60 seconds for those jobs
and their server health endpoints to stop. `SERVER_CLEANUP_COMPLETE` precedes
return to the smoke wrapper. A cleanup timeout returns failure; inspect it before
starting another benchmark. A pre-existing service is still rejected, never killed.

```bash
bash scripts/smoke_stateful_memory_qwen35_9b_4node_idev.sh discoveryworld agentfold
bash scripts/smoke_stateful_memory_qwen35_9b_4node_idev.sh discoveryworld supo
```

For SWE-bench Lite:

```bash
bash scripts/smoke_stateful_memory_qwen35_9b_4node_idev.sh swe-lite agentfold
bash scripts/smoke_stateful_memory_qwen35_9b_4node_idev.sh swe-lite supo
```

For ScienceWorld (Java must be on PATH, or set `JAVA_HOME`):

```bash
bash scripts/smoke_stateful_memory_qwen35_9b_4node_idev.sh scienceworld supo
```

Each command defaults to two tasks split across two server/evaluator pairs. The
DiscoveryWorld catalogue is all difficulties (120 tasks in full). Vista SWE uses
the pinned ARM image-available Lite subset, not the official 300-task x86 suite.
Two-task smoke samples do not cover all difficulties or all repositories.

The existing Docker runner also accepts `--method agentfold` and `--method supo`
on an x86 host. Its official harness grading remains a separate step.

## Saved sbatch entry

From a login node, submit the same two-task smoke with a six-hour, four-node job:

```bash
sbatch scripts/eval_stateful_memory_qwen35_9b_4node.sbatch discoveryworld agentfold
sbatch scripts/eval_stateful_memory_qwen35_9b_4node.sbatch discoveryworld supo
sbatch scripts/eval_stateful_memory_qwen35_9b_4node.sbatch swe-lite agentfold
sbatch scripts/eval_stateful_memory_qwen35_9b_4node.sbatch swe-lite supo
sbatch scripts/eval_stateful_memory_qwen35_9b_4node.sbatch scienceworld supo
```

`SAMPLES=-1` selects the full catalogue/subset after smoke validation. Six hours
is the request limit, not a guarantee of completion, especially for SWE image
preparation and test execution. `STATEFUL_PROJECT_ROOT` can point the sbatch entry
to a pinned checkout; Slurm's copied script is never used to locate repository files.

The `stateful-memory-smoke-audit.json` report requires both shards complete,
native grading available, trajectories/provenance present, tools and memory
exercised, and no terminal format failures. It does not require a correct answer
or resolved patch. A task that finishes without needing memory is valid but does
not by itself validate memory mechanics. Raw per-request and per-tool logs remain
in the native runner layouts. Do not pool these runs with BC-P or earlier adapters.

ScienceWorld audits also show the last six commands, bounded raw observations,
scores and terminal flags, omitting the simulator's large `valid` action lists.
If `memory_exercised` is false after an early task failure, the memory path remains
untested; neither a prompt change nor a completed grading result establishes it.
All native audits include the last three format errors and bounded response tails.

### ScienceWorld SUPO memory-path diagnostic

ScienceWorld observations can stay below the normal 32K summary threshold for the
entire episode. A graded run with no summary does not test summary recovery. Use
the dedicated two-task diagnostic to lower only the summary threshold to 4096:

```bash
bash scripts/smoke_scienceworld_supo_memory_4node_idev.sh
```

This retains 64K context, the cumulative token budget, focus_v3 and environment
scoring. Config records `evaluation_scope=memory_smoke_4k_not_benchmark`, and
trajectory provenance records the actual threshold. The summary itself can still
fail or a task can end before the threshold; the strict audit remains unchanged.
These scores are not comparable to the regular 32K-threshold benchmark. The
ordinary launcher still uses 32K; do not lower it to improve full-run smoke status.
