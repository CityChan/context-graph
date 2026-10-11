# ScienceWorld GraphRPO v3_long

This opt-in training profile implements the objective described in the supplied
ScienceWorld v3_long screenshots. It is **not a verified reproduction of the
unavailable `huanchen/graphrpo-compress-rl` branch**. The backbone and deployment
hardware must be selected explicitly. Existing benchmark evaluation and ordinary
GraphRPO controller configurations remain available for historical runs. This
profile now defaults to model-owned explicit graph tools, a protocol change from
its initial checkpoint-controlled version; do not pool their results.

## Model-owned graph tools

`graph_action_protocol: explicit_tools` now uses `typed_action_json_v1` on every
main and branch action turn. The model chooses a tool and typed arguments under
vLLM JSON-schema constrained decoding; the framework serializes the validated
decision for existing environment adapters. The model does not write XML.
This replaces the earlier unconstrained explicit-XML profile; keep results separate.
The model chooses whether and when to call graph tools:

| Tool | Effect |
| --- | --- |
| `merge(node_ids, summary)` | Fold 2–6 active nodes into a model-written summary. |
| `prune(node_id)` | Remove a non-root node from active memory; retain raw evidence. |
| `add_edge(source, target, relation)` | Add a causal, semantic or temporal edge. |
| `select(node_id)` | Set graph focus. |
| `graph_read(node_id?, offset?)` | Read stored evidence, including folded/pruned nodes; omit ID to page through all IDs. |
| `recall(node_id)` | Restore an inactive node. |
| `graph_compact()` | Rebuild working context from the original task and **all** currently active nodes and edges. |

Merge/prune change the graph immediately. Existing conversation turns remain
visible until the model calls `graph_compact`; that call makes the chosen memory
affect subsequent model input. An oversized snapshot is rejected, with feedback
to merge/prune further. The framework never silently selects nodes to drop.
Read pages contain up to 2,000 characters or 20 node IDs; offsets expose the rest.

There are no forced graph checkpoints, automatic prune/merge/semantic links,
automatic session summaries, or automatic branch archives in this protocol.
Observation records and temporal/delegation links still record executed actions.
Branches start from the original task plus their assignment and shared graph
state, and can read/edit the same graph. Returning adds their report without
folding or deleting their observations. The ten-operation graph quota is disabled;
episode turns, token budgets and branch limits still apply.

Each explicit context reset exports a separate PPO segment with the exact prefix
used for generation. Prior policy tokens/log probabilities remain immutable and
trainable; no branch or reset segment is sampled away by `max_traj`. Branch work
and graph refresh payloads consume the episode budget without refunds. Invalid
graph calls receive token-local `-0.3`; graph decisions retain their span markers.
Under this profile's `outcome_only` backend, both main and branch graph tools learn
from shared episode advantage and process penalties; local graph delta remains
zero. Other existing local-credit backends still score main-agent edits only,
and branch events must never be mapped onto a main-agent turn index.

Other isolated-agent configurations can opt in with the same
`actor_rollout_ref.rollout.plugin.graph_action_protocol=explicit_tools` field
(use Hydra `+` when adding a field absent from that configuration). Explicit mode
overrides old automatic-maintenance flags in a per-rollout copy. Existing E/M
continuation uses controller checkpoints and rejects explicit mode. Non-GraphRPO
benchmark/memory-baseline launchers retain their existing protocol unless opted in.

### Action protocol and training records

The supported typed interfaces are BC-P/local GAIA, SWE, ScienceWorld and
DiscoveryWorld. Every ordinary action, branch action and finalizer call supplies
an output schema. Templates use `enable_thinking=false`: this action-only profile
does not emit a separate reasoning block. A branch has `return` instead of
`branch`/`finish`; simulator episodes have no model finish tool. DiscoveryWorld
uses its public action catalogue and dialog state to constrain the command object
and argument types, without consulting grading information. No action is forced
merely to exercise a tool. Target existence and other semantic preconditions are
still checked by the tool/environment after syntax validation.

JSON parsing rejects partial objects, duplicate keys, unsupported tools and
argument types before dispatch. Legacy transport delimiters embedded in arguments
are rejected rather than interpreted as extra calls. Budget truncation can still
interrupt a structured generation; it is recorded as an action protocol error,
never completed by guessing fields. The finalizer restricts the schema to finish
and does not synthesize a fallback answer from rejected output. These checks do
not guarantee that the chosen action is useful or that the task will succeed.

Raw sampled JSON, IDs and log probabilities remain in the model history and PPO
segments. The XML execution representation is stored separately in per-turn
metadata; it never replaces sampled tokens or receives their probabilities.
Graph event hashes refer to the raw policy decision, and process labels use the
dispatched action. The existing PPO objective recomputes old/new raw-model
likelihoods on these sequences; this is constrained rollout collection, not an
implementation of grammar-renormalized PPO. Mixing backend guided log-probs into
the PPO denominator or rollout importance/rejection correction is rejected.
The v3_long profile already uses recomputed old-policy probabilities.

Manifests record the output profile and thinking setting, and trajectories expose
`constrained_action_protocol` and `action_protocol_errors`. BC-P also probes the
full action schema before starting tasks. CPU tests verify schema validity,
dispatch, role restrictions, malformed-output rejection and exact training token
alignment; live vLLM grammar execution still needs the GPU preflight/evaluation.

## Objective

Let `c` be native task completion, `p` the highest environment score observed in
the episode, and `natural_end` an environment termination other than its step
cap or an infrastructure error. The training terminal reward is:

| Outcome | Training reward |
| --- | --- |
| Completed | `1` |
| Naturally ended, incomplete (including wrong focus) | `0.15 * clip(p, 0, 100) / 100` |
| Incomplete, token cap or five consecutive action turns without a parsed function call | `-0.1` |
| Other incomplete truncation: turn cap, timeout, environment step cap | `0` |
| Environment infrastructure failure | `0`, with error telemetry retained |

The no-tool counter spans main and branch action turns, resets on a parsed
function call, and excludes JSON controller turns. A cap stops the whole episode
without a final synthesis call. A malformed/unknown call still receives the
existing validation feedback/process penalty; this cap does not redefine the
legacy function-call parser. A negative terminal environment score identifies
natural failure even if it coincides with the adapter step cap.

Validation keeps the native **binary completion reward**, using the same
`legacy` prompt and no-tool stopping rule as training. Rollouts retain raw
`environment_score`, `environment_peak_score`, `environment_natural_end`, native
`task_reward`, and separate `training_reward` (training only). Exported rows also
record `native_task_reward`, `training_reward_applied`, the reward profile and
credit backend. Partial training reward must not be reported as benchmark success.

The ordinary GraphRPO advantage is

`A(q,g,t) = R(q,g) - mean_g R(q,g) + 0.1 * d(g,t) + 0.25 * p(g,t)`.

Main and branch streams are deduplicated by episode ID before computing the
mean. `graphrpo_norm_adv_by_std=false` disables standard-deviation normalization;
the global default remains `true`, and alternating E/M objectives are unaffected.
The explicit `outcome_only` backend sets local graph credit `d` to zero and
does not request evaluator/probe calls. Existing token-local process labels
remain: tool formatting/no-call errors `-1`, unfinished final completion `-1`,
unfolded-main task turns `-1`, and invalid graph decisions `-0.3`. Overlapping
labels use the existing minimum precedence, then clamp to `[-1,0]`. This profile
does not enable compress or scope penalties.

| Setting | Value |
| --- | --- |
| Initial questions per batch / episodes per question | `8 / 4` |
| Dynamic group filter | outcome variation or negative process label |
| Minimum informative groups / maximum generation batches | `4 / 3` (initial plus at most two extras) |
| Advantage normalization | subtract episode-group mean only |
| Token weight | `1 / (number of retained questions * episodes per question * trainable episode tokens)` |
| PPO clip low / high | `0.2 / 0.28` |
| KL coefficient / estimator | `0.0005 / low_var_kl`, same episode weights |
| Learning rate / actor gradient clip | `2e-6 / 0.5` |
| Optimizer epochs | `1`; GraphRPO uses one minibatch containing all retained streams |
| Rollout-log-prob override | disabled; recomputed raw-policy likelihood on sampled JSON |
| Overlong masking | disabled; preserve truncation penalties |

Dynamic sampling can retain fewer than four groups when its three-batch budget
or the epoch iterator is exhausted; the existing driver reports this. An empty
selection skips optimization. Nonfinite reward/advantage checks and synchronized
loss protection remain enabled.

## Running

The complete Hydra configuration is
`verl/trainer/config/scienceworld_graphrpo_v3_long.yaml`.
The following are explicit operational defaults, **not recovered screenshot
settings**: 65,536-token context (`8,192 + 57,344`), 100 total turns, 100
environment steps, 2,048 tokens per action, up to five branches, 8,192 branch
tokens, model-selected graph-tool timing, 100 training steps, validation
and checkpoint every ten steps. All main/branch streams are retained. Training
and validation share the response budget and prompt. Graph and environment tools
use the same action-policy sampling: temperature `1`, top-p `1`.

Use an existing training environment with the required VERL/vLLM model support,
ScienceWorld and Java. This launcher does not install dependencies or change
environments. Existing `scripts/make_scienceworld_data.py` produces the expected
`data/scienceworld_graph_train.parquet` and `data/scienceworld_graph_val.parquet`
from the official train/dev splits. The test split is not used for training.

From the repository root in **Bash**, resolve and save the protocol without
importing the GPU trainer or starting Ray:

```bash
MODEL_PATH=/path/to/intended/checkpoint DRY_RUN=true bash scripts/train_scienceworld_graphrpo_v3_long.sh
```

After providing the checkpoint/data and a suitable GPU allocation, launch:

```bash
MODEL_PATH=/path/to/intended/checkpoint bash scripts/train_scienceworld_graphrpo_v3_long.sh
```

`PYTHON`, `TRAIN_FILE`, `VAL_FILE`, `RUN_DIR`, `GPUS_PER_NODE` and `NNODES` are
optional environment overrides. Extra command arguments are Hydra overrides,
for example `trainer.total_training_steps=2 trainer.test_freq=1`. By default
this uses **one local GPU**, without allocating nodes or stopping other jobs.
For multiple nodes, supply an already configured Ray cluster using VERL's Ray
configuration in addition to `NNODES`; setting `NNODES` alone does not start it.
Memory capacity and model/context compatibility need validation on that hardware.

Every invocation writes a timestamped resolved `protocol-*.yaml` and a JSON
record containing the commit, dirty paths, Python executable, command overrides
and reward semantics under the run directory. Subsequent rollout and validation
artifacts go there too. `DRY_RUN` checks/configures the protocol only; it does not
validate checkpoint weights, datasets, JVM startup, CUDA or distributed execution.

## Zero-shot evaluation on existing Vista idev nodes

BC-P and DiscoveryWorld accept `GRAPH_ACTION_PROTOCOL=explicit_tools` with
`contextgraph`. This uses the Qwen3.5-9B actor for explicit graph decisions and
disables automatic pruning, branch archival, semantic edge binding and scheduled
graph maintenance. Model calls to graph tools remain enabled. Omitting the flag
preserves the existing `controller` evaluation protocol.

From a clean, updated checkout inside an idle **four-node** allocation, run a
small BC-P evaluation:

```bash
GRAPH_ACTION_PROTOCOL=explicit_tools SAMPLES=3 DATA_PATH="$PWD/data/bc_test.parquet" bash scripts/eval_bcp_qwen35_9b_4node_idev.sh contextgraph
```

For two DiscoveryWorld tasks at the default Normal difficulty, using the same
compact observation profile as the memory integration smoke:

```bash
GRAPH_ACTION_PROTOCOL=explicit_tools BENCH_SAMPLES=2 BENCH_DISCOVERYWORLD_OBSERVATION_PROFILE=compact_v1 bash scripts/eval_discoveryworld_qwen35_9b_4node.sbatch contextgraph
```

Run these sequentially in one allocation, or in separate idle allocations. Both
launchers manage their own model servers. `SAMPLES=-1` and `BENCH_SAMPLES=-1`
select the full respective dataset. Existing context defaults remain 32K for
BC-P and 64K for DiscoveryWorld; match budgets and observation profiles when
comparing against trained policies.

The launch log prints `GRAPH_ACTION_PROTOCOL=explicit_tools`. Each evaluation
manifest records the effective protocol and disabled maintenance settings under
`config.actor_rollout_ref.rollout.plugin`; DiscoveryWorld forwards the protocol
to every task worker. Local tests verify configuration, worker parsing and shell
argument forwarding. Live idev startup and model graph-tool usage still require
inspection of the resulting logs and trajectories.

For an existing BC-P output directory, inspect the actual stop reasons, effective
configuration, model graph-tool attempts/successes, trace event sources and last
main/branch responses without rerunning inference:

```bash
python scripts/diagnose_explicit_graph_eval.py /path/to/bcp-output
```

This diagnostic does not assign a pass flag. A zero tool count means the model
did not exercise that explicit interface in the recorded trace; an absent trace
is reported separately. Normal evaluation completion and generation-quality
checks alone do not establish successful graph usage or task performance.

In explicit mode, branches share the episode token allowance and cannot spend
the main agent's configured final-answer reserve and safety margin. Branch
generations and visible tool observations must fit their assigned allowance;
oversized observations are truncated using the existing head/tail budget helper.
This does not force a graph operation, grant extra tokens or enable automatic
graph maintenance. Environment-terminated benchmarks with a zero final-answer
reserve keep that setting.

Older `model_contexts` records contain input messages and a `max_len` ceiling
only. The diagnostic reports missing response fields as unknown and includes
the last visible input messages and saved transcript tail. New records also
capture the actual response and output token IDs. A budget-triggered finalizer
completion should be distinguished from a natural model finish using
`natural_finish`, `forced_finish` and `pre_finalize_token_limit`.

## Verification boundary

Local regression coverage exercises reward precedence, peak-score reset,
natural termination versus step cap, real executor paths with a scripted
simulator/model, main/branch no-tool stops, native validation rewards, prompt
matching, dynamic filtering, config composition, unchanged E/M baselines and
episode-deduplicated mean centering, explicit main/branch graph edits, retained
raw evidence, absence of automatic maintenance, and exact policy-prefix alignment
across model-selected context resets. No live GPU training or model-performance
claim follows from these tests.
