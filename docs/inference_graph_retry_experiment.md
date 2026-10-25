# Inference graph retry experiment

Branch: `experiment/inference-graph-retry-guards`, based on master `88d7423`.
The feature is off unless `plugin.graph_retry_guard=true`. Training rejects this
option. Existing auto/controller timing, task prompts, thinking settings,
compression schedules and global turn/time limits remain the comparison controls.

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
