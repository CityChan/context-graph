# AgentFold and SUPO on SWE-bench Lite and DiscoveryWorld

Both are evaluation-only adaptations using Qwen3.5-9B, not the papers' trained
policies. BC-P/GAIA retain the validated v4 behavior. No live GPU run for these
new environments has been completed by the implementation tests.

## Native environment contracts

AgentFold keeps the same model-selected contiguous suffix folding and strict
whole-block validation. Only the tool contract changes: `python_exec(code)` and
`finish(message)` for SWE, or `action(command)` for DiscoveryWorld. No local
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

## SUPO stateful overflow adaptation

BC-P continues to exclude a threshold-crossing round from working memory, as
before. In these stateful environments an executed edit/move cannot be forgotten
without losing the current state. The `retained_round_v1` profile instead summarizes
the preceding context and appends the actual crossing call and its shown observation
to the resumed context. Actions are never replayed or rolled back. Observations
are charged once and any budget truncation is explicit. If that retained round
plus summary cannot fit below the threshold, the adapter fails visibly; it never
silently drops current state. Provenance records this deviation from the paper.

Both environments use 64K context in the supplied launcher. SUPO defaults to
a 32K working-context threshold, two summaries and 1024 output tokens per summary.
The bounded summary format retains the existing v4 character cap. Summary requests
consume model turns, tokens and time but do not advance the simulator or execute
repository commands. Memory compression does not refund cumulative token budget.
DiscoveryWorld allows 200 model turns and at most 200 environment steps. SWE uses
the existing runner's 100-turn default. These choices are recorded in task config.

## Four-node Vista smoke

Run one method in each allocation, or run them sequentially after the previous
launcher exits. Outputs are created under the checkout's `output/` directory;
inherited old run directories are ignored. The dedicated SWE environment and its
preparation/grading sub-environments must already exist; `SWE_AGENT_ENV` can select
the previously validated installation. DiscoveryWorld uses the existing private
Python 3.10 overlay selected from `cxtgraph`.

```bash
bash scripts/smoke_stateful_memory_qwen35_9b_4node_idev.sh discoveryworld agentfold
bash scripts/smoke_stateful_memory_qwen35_9b_4node_idev.sh discoveryworld supo
```

For SWE-bench Lite:

```bash
bash scripts/smoke_stateful_memory_qwen35_9b_4node_idev.sh swe-lite agentfold
bash scripts/smoke_stateful_memory_qwen35_9b_4node_idev.sh swe-lite supo
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
