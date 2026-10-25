# Inference graph retry experiment

Branch: `experiment/inference-graph-retry-guards`, based on master `88d7423`.
The feature is off unless `plugin.graph_retry_guard=true`. Training rejects this
option. Existing auto/controller timing, task prompts, thinking settings,
compression schedules and global turn/time limits remain the comparison controls.

The smoke wrapper now sets the same bounded diagnostic limits in both arms:
12 turns, a 600-second cooperative session limit, a 900-second asyncio episode
deadline covering rollout and grading, and (controller mode only) a three-turn
no-parsed-tool limit shared by main and branches. `EVAL_MAX_TURN`,
`EVAL_SESSION_TIMEOUT`, `EVAL_EPISODE_TIMEOUT`, and `EVAL_NO_TOOL_LIMIT` override
these values; the ordinary evaluator defaults remain 100/3600/0/0. Async deadlines
depend on cooperative cancellation, not an OS process kill. A timed-out episode
is an execution error and retains its request log, not a valid completed trace.
Both arms use the same limits and thinking settings. These short runs are not
directly comparable with the original full-length A/B. A zero-task-tool baseline
fails before launching guarded; inspect existing artifacts first.

Controller smoke also defaults `EVAL_ACTION_RECOVERY=1` in **both** arms. If an
ordinary task response has no complete, role-appropriate native tool call, the
next ordinary turn is a single no-thinking, regex-constrained retry capped at
512 output tokens. The policy still chooses search/open_page/branch/finish (main)
or search/open_page/return (branch). A failed constrained retry terminates with a
format error; it is never dispatched. Finalization uses the same no-thinking
constraint restricted to finish. Successful ordinary turns keep their thinking
settings. Exact retry inputs/outputs are captured, and retries consume turn,
window and request accounting. This is inference-only; training and explicit
graph policy paths do not enable it. Set `EVAL_ACTION_RECOVERY=0` to reproduce
the earlier task-output behavior. This shared actor change means new runs cannot
be directly compared with the earlier raw baseline as an isolated guard effect.

Recovery arguments have no per-field character ceilings. The earlier 384-character
branch prompt, 512-character return and 256-character explanation caps could force
closing tags in the middle of a sentence, even with `finish_reason=stop`. The
512-token request cap remains; the instruction asks for concise, complete arguments.
An incomplete XML response at the token cap is rejected, not repaired or dispatched.
Syntactic acceptance still does not establish that a delegation or return preserves
all relevant evidence; inspect the saved text before drawing quality conclusions.

The guarded arm adds dynamic node/edge constraints, action-specific controller
arity, rejection accounting, short error feedback, and a failure cache based on
target state (not counters). After three consecutive failures, explicit tools
temporarily exclude graph tools until a real task tool executes. A successful
read, compaction or controller pass cannot clear this cooldown. Main and explicit
branches share it. A relevant target change allows retrying a cached failure;
an unrelated observation does not. Invalid XML graph calls outside controller
mode also consume the controller graph-attempt budget.

Graph schemas still cannot enforce every semantic constraint. In particular,
merge uniqueness is validated at execution because the deployed backend does not
reliably implement uniqueItems. A rejected graph_compact is not cached by graph
state alone, because its preconditions depend on the current rendered transcript.
No new lifetime compute cap or automatic edit timing policy is introduced.
This first patch does not yet detect every legal-but-unproductive read/compact
loop, and does not change SFT or RL policies.

## Four-node Vista smoke

Before repeating the graph A/B, use
`bash scripts/smoke_bcp_action_recovery_4node_idev.sh` inside the experiment
worktree. It selects original BC-P indices 28 and 70, runs one guarded arm with
shared actor recovery enabled, and limits each episode to six normal turns
(plus the existing finalizer reserve). It reports `task-recovery-audit.json`;
success requires real task tools on both questions and at least one valid
constrained retry. This verifies recovery mechanics, not answer accuracy or a
benefit from graph guards. Startup/model loading is still required once.

Use a separate checkout so running master jobs never import modified files:

```bash
cd /work/09281/chc_1996/vista/context-graph
git fetch origin experiment/inference-graph-retry-guards
git worktree add --detach ../context-graph-retry-smoke origin/experiment/inference-graph-retry-guards
cd ../context-graph-retry-smoke
SAMPLES=3 bash scripts/smoke_graph_retry_ab_4node_idev.sh
```

Run inside the existing four-node idev. Defaults are Qwen/Qwen3.8-27B, BC-P,
64K rendered working context, repaired memory, controller protocol, seed 42,
one concurrent episode per server. An existing local scratch model snapshot is
required. Override MODEL_ID for Qwen/Qwen3.5-9B or Qwen/Qwen3.5-27B.
MODEL_PATH is preserved when supplied. Without it, the resolver uses the selected
model's cached refs/main, or its sole complete snapshot if refs/main is absent.
Multiple complete snapshots require an explicit MODEL_PATH. It never downloads
weights or silently selects a different model; missing caches list repository
directories for diagnosis. A path inside another model's HF cache is rejected.
Set GRAPH_ACTION_PROTOCOL=explicit_tools to test model-owned timing separately;
do not combine results from different protocols.

The runner starts one retrieval server and three model servers once, then runs
baseline and guarded arms with the same task selection. Each arm has its own
manifest, request logs, trajectories and results. All outputs live in a fresh
scratch directory, never in a prior run. The dataset is read from the original
checkout by default. If the worktree already exists, inspect it instead of
overwriting it or changing its revision during a run.

The audit requires paired IDs and matching manifests except the guard flag,
complete artifacts, 64K evaluator settings and successful execution/judging.
It reports guard/cooldown exercise separately: zero failures can be a valid run,
but does not verify recovery on real model failures. Baseline graph_invalid_ops
misses structured-controller failures; compare controller_errors and traces too.
The guarded arm has warm caches, so wall-clock latency is not a controlled metric.
Three questions do not establish an accuracy gain.

## Diagnose saved zero-tool episodes

`scripts/diagnose_bcp_stalled_actions.py RUN_ROOT` reads the saved baseline
trajectories for indices 28 and 70, reports finish reasons/output-limit counts,
and prints bounded first/last response samples plus preceding feedback. It uses
only Python's standard library and never loads a model or makes API calls.
Override `--indices` or `--variant` for other saved episodes. Zero-tool episodes
now make the paired audit fail even when process status is `ok`. Guard invocation
and actual failure/recovery exercise are reported separately.
