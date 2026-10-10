# ScienceWorld GraphRPO v3_long

This opt-in training profile implements the objective described in the supplied
ScienceWorld v3_long screenshots. It is **not a verified reproduction of the
unavailable `huanchen/graphrpo-compress-rl` branch**. The backbone and deployment
hardware must be selected explicitly. Existing benchmark evaluation and ordinary
GraphRPO defaults remain unchanged.

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
| Rollout-log-prob override | disabled; existing on-policy actor path |
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
tokens, graph checkpoint every five main turns, 100 training steps, validation
and checkpoint every ten steps. All main/branch streams are retained. Training
and validation share the response budget and prompt. Controller sampling uses
temperature `1`, matching action sampling, and top-p `1`.

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

## Verification boundary

Local regression coverage exercises reward precedence, peak-score reset,
natural termination versus step cap, real executor paths with a scripted
simulator/model, main/branch no-tool stops, native validation rewards, prompt
matching, dynamic filtering, config composition, unchanged E/M baselines and
episode-deduplicated mean centering. No live GPU training or model-performance
claim follows from these tests.
