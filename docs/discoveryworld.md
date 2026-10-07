# DiscoveryWorld paired evaluation

This integration runs the **real** [AllenAI DiscoveryWorld simulator](https://github.com/allenai/discoveryworld), pinned to `fd591323920be0d3786ef350955de1945aa571e5`, with ContextGraph and FoldAgent. It is separate from DiscoveryBench's dataset analysis tasks.

## Protocol

- Default: eight official scenarios, Normal difficulty, seeds 0–4 (40 public tasks). `BENCH_DIFFICULTY=all` selects all three difficulties (120 tasks). No invented train/test split, tutorials, or small-skill tasks.
- Fixed Qwen3.5-9B checkpoint, deterministic decoding, thinking enabled, same context/action/time budgets for both methods. Defaults: 65,536-token context, 100 task turns, 57,344 generated-token budget, 3,600-second session timeout. Graph controller calls use tokens/time but do not consume task turns with the default repaired memory profile. Branches and invalid tool calls still use task turns; only actual submitted environment actions tick the world. Set `BENCH_MAX_STEPS` explicitly for longer experiments.
- Text-only: the official `UserInterface.renderJSON()` component of `getAgentObservation()`; image rendering and PNG saving are skipped. Public inventory, accessible objects, dialogs, feed, task description and action feedback are retained. No global object enumeration or hidden hypotheses enter the model prompt.
- XML `action` tool with JSON in `command`; official action names and UUID arguments. During dialogs use `chosen_dialog_option_int`. Every accepted command calls `performAgentAction` followed by exactly one `tick`, including unsuccessful environment actions. Graph and branch operations do not tick the simulator.
- Success is the official `completedSuccessfully`, not merely a terminal state. Report the official normalized procedural score ×100 and task success separately. **The paper's knowledge-discovery metric is not implemented**; `knowledge_score` is null. These are text-only paired agent results, not reproduction of the full three-metric paper protocol.
- Full oracle scorecards are written to `scorecard.json` for grading only. Public `tools.jsonl`, raw `requests.jsonl`, trajectory, immutable manifest, package versions and simulator provenance are saved separately. Existing generation-audit criteria remain unchanged; passing them does not exclude semantic repetition.
- One child process per task. Simulator/model errors are infrastructure errors, not model score zero. Resume requires identical code/config/data/environment; use `BENCH_RETRY_ERRORS=1` only to retry infrastructure errors.

## Optional compact public observations

`BENCH_DISCOVERYWORLD_OBSERVATION_PROFILE=compact_v1` (CLI:
`--discoveryworld-observation-profile compact_v1`) enables the same encoding for
both agents. The default is `full`, retaining the existing JSON serialization.
This setting applies only to DiscoveryWorld; it does not change ScienceWorld,
the common token/turn accounting, simulator actions, or scoring.

Each compact observation is a self-contained snapshot. Inventory, accessible
objects, and nearby objects use column/row tables. Repeated long strings in those
tables refer to a dictionary included in that same observation. The embedded
legend distinguishes dictionary indices from object UUIDs. Directions, distances,
UUIDs, names, descriptions, ordering, duplicates, and all other public fields are
retained. Task descriptions, dialogs, measurements, and action results remain
present on every applicable step. There is no dependence on a previous snapshot
and no new oracle information. Unknown or nested object records stay verbatim.

The selected profile is recorded in the manifest/config and compact result
protocol; resume and paired audits reject mismatched profiles. `tools.jsonl`
continues to store the original public UI. The model receives the encoded UI
both at reset and after actions. Use a new run directory when changing profiles.

Audit existing raw logs without starting a simulator or model:

```bash
python scripts/audit_discoveryworld_observations.py "$DW_RUN" --tokenizer "$SCRATCH/hf_cache/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a" --output "$DW_RUN/compact-observation-audit.json"
```

The audit checks JSON round-trip equality and measures actual tokenizer lengths,
including the compact legend and text dictionary. Counts cover serialized public
observations/action results, not complete prompts, action catalogues, model
generations, or future behavior. Without `--tokenizer`, it reports character counts
only. It refuses to overwrite an existing report.

Local validation on the pinned simulator, Normal difficulty, seed 0, all eight
scenarios (reset plus two rotation actions each) reconstructed all 24 snapshots.
Using the pinned Qwen3.5-9B tokenizer, per-scenario serialized observation token
reductions were approximately 27–32%. This is a short mechanics/encoding probe,
not a replay of the Vista trajectories or evidence of improved task success.

## Vista four-node smoke

Use a clean committed checkout. The launcher selects an existing Python 3.10/3.11
agent interpreter, preferring `cxtgraph`. The version probe uses `-I -S` to skip
site initialization, `.pth` hooks, and package scanning; a timeout means the version
is unknown, not incompatible. It builds a private overlay and does not modify
`cxtgraph`, `deepseek_v4`, or running evaluations. An explicit `BENCH_BASE_PYTHON`
override must pass the same checks; an incompatible override fails without fallback.
Pinned upstream uses float-to-integer conversions removed in Python 3.12.

The selected interpreter and overlay path are printed before setup. Default
overlays now end in `-py310` or `-py311`, avoiding reuse of the earlier unversioned
overlay that could have been created with the SWE environment's Python 3.12.
Existing overlays are also checked before pip or model startup. No old environment
is deleted or rewritten to change its Python version. If no compatible interpreter
is confirmed, setup stops and lists the probe failures. After installation, the
private environment undergoes normal Python startup, real agent-package imports,
and simulator checks before model servers start. This separates interpreter-version
selection from dependency and simulator validation.

On an allocated four-node `idev`, from the checkout:

```bash
env -u BENCH_RUN_DIR -u BENCH_DATA -u BENCH_AGENT_ENV PROJECT_ROOT="$PWD" BENCH_SAMPLES=2 BENCH_MAX_STEPS=100 BENCH_DIFFICULTY=Normal bash scripts/eval_discoveryworld_qwen35_9b_4node.sbatch both
```

`both` assigns one server/evaluator pair to each method, evaluating the same two tasks (Chemistry and Archaeology, seed 0). Before model startup, a real two-action simulator probe runs all eight requested scenarios at seed 0. This is an infrastructure check, not a success-rate test.

For a more useful memory smoke use `BENCH_SAMPLES=8` (one task per scenario), and inspect controller calls/turn limits before a full run. Two tasks do not establish comparative performance.

## Full suite submission

From a clean committed checkout on a Vista login node:

```bash
bash scripts/submit_discoveryworld_qwen35_9b_all200.sh
```

This submits two independent sbatch jobs, one per method. Each requests four
nodes for 48 hours, with two server/evaluator pairs splitting 120 tasks into
60 tasks each. The fixed protocol is all eight scenarios, all three difficulties,
five seeds, `compact_v1`, 65536 context tokens, 200 environment/agent turns,
repaired memory, and eager serving. The token budget remains unchanged.

The submitter creates a detached worktree at the invoking checkout's commit and
separate fresh run directories. Old `BENCH_DATA`, `BENCH_AGENT_ENV`, and run-directory
settings do not redirect this submission. It records the commit, both job IDs,
artifact paths and log paths under
`$SCRATCH/context-graph-agent-benchmarks/submissions/discoveryworld-all200-*/`.
The exact directory is printed as `DISCOVERYWORLD_SUBMISSION`; `jobs.tsv` is
updated after each successful submission. If the second submission fails, the
first job remains submitted and recorded; inspect that record before retrying.
Existing jobs are never cancelled. Running the submitter again creates new jobs.

For only the Normal suite, the configurable evaluator remains available:

```bash
mkdir -p logs
env -u BENCH_RUN_DIR -u BENCH_DATA -u BENCH_AGENT_ENV PROJECT_ROOT="$PWD" BENCH_SAMPLES=-1 BENCH_MAX_STEPS=200 BENCH_DIFFICULTY=Normal sbatch scripts/eval_discoveryworld_qwen35_9b_4node.sbatch both
```

Single-method `contextgraph` or `foldagent` uses both pairs as disjoint shards. Run outputs are under `$SCRATCH/context-graph-agent-benchmarks/runs/discoveryworld-9b-...`; the launcher prints the exact path and captures `suite.log`, including direct idev launches.

```bash
cat "$DW_RUN"/contextgraph*/summary.json "$DW_RUN"/foldagent*/summary.json
tail -n 10 "$DW_RUN"/evaluator-*.log
python scripts/audit_discoveryworld_pair.py --contextgraph "$DW_RUN" --foldagent "$DW_RUN" --output "$DW_RUN/pair-audit.json"
```

The audit compares only common graded task IDs and checks model, revision, code, data, budgets and config compatibility. It refuses overwriting an earlier audit.

## Local verification

In an isolated Python 3.10/3.11 agent environment:

```bash
python -m pip install -r requirements_discoveryworld.txt
python scripts/probe_discoveryworld.py --output output/discoveryworld-simulator-probe
DISCOVERYWORLD_LIVE_TEST=1 SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy python -m pytest tests/test_discoveryworld.py -q
```

The opt-in tests exercise all eight real Normal scenarios and both real agent loops with a scripted model client. They verify integration, not Qwen performance or Vista GPU compatibility. No language-model or paid judge API is used by the simulator probe.
