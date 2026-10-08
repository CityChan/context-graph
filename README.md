# ContextGraph

ContextGraph studies graph-structured working memory for long-horizon agents.
This repository contains ReAct, FoldAgent and ContextGraph executors, benchmark
environments, evaluation audits, and a modified `verl` training stack.

**Current status:** saved BC-P and local-GAIA evaluations are available with
audited provenance. SWE-bench has a one-task ARM compatibility record, not a
full official score. The five-node Qwen3.5-9B RL launcher is implemented;
successful 32K training updates and memory fit on that topology remain unverified.

## Start here

| Task | Entry point / guide |
| --- | --- |
| Understand the modules and experiment variants | [Codebase map](docs/codebase_map.md) |
| Review confirmed fixes and remaining risks | [October 1 review](docs/project_review_20261001.md) |
| Train Qwen3.5-9B on Vista | [RL setup, preflight, smoke and submission](docs/bcp_qwen35_9b_rl.md) |
| Train executor and memory roles with EM-RPO | [Shared-policy training, compatibility and limitations](docs/em_rpo.md) |
| Evaluate BC-P / text-only local GAIA | [BC-P guide](docs/bcp_qwen35_9b_eval.md), [GAIA guide](docs/gaia_qwen35_9b_eval.md) |
| Smoke-test MemoBrain and A-MEM on BC-P | [Protocol and four-node idev commands](docs/graph_memory_baselines.md) |
| Evaluate SWE-bench Verified or Lite | [Container, model and grading guide](docs/swebench_verified_eval.md) |
| Submit Lite ARM containers + inference + grading in one job | [Four-node batch entry](docs/swebench_verified_eval.md#all-in-one-four-node-batch-submission) |
| Re-evaluate DiscoveryBench with Qwen3.5-9B | [Paired ContextGraph/FoldAgent batch job](docs/discoverybench_qwen35_9b_eval.md) |
| Evaluate ScienceWorld with Qwen3.5-9B | [Four-node setup, metrics, trajectories and resume](docs/scienceworld_eval.md) |
| Run GRAM document-QA or its BC-P adaptation | [Four-node BC-P smoke, graph protocol, evaluation and GRPO training](docs/gram.md) |
| Run SAB, ALFWorld or teacher-data workflows | [Additional workflows](docs/workflows.md) |
| Find other runnable scripts | [Script index](scripts/README.md) |
| Inspect saved results | [Result index](results/README.md) |

## Results and evidence

The following historical Qwen3.5-9B runs use a 32,768-token working-context
limit. Counts were recalculated from per-instance records and checked against
the saved shard manifests and summaries. The [audit bundle](results/audited/README.md)
contains exact run IDs, commits, checkpoint revision, selected indices, settings,
source hashes, per-instance outcomes and excluded runs.

| Benchmark | ReAct | FoldAgent | ContextGraph |
| --- | ---: | ---: | ---: |
| BrowseComp-Plus, 150 questions | 65/150 (43.33%) | 68/150 (45.33%) | 74/150 (49.33%) |
| GAIA text-only, local BC-P retrieval, 127 questions | 9/127 (7.09%) | 12/127 (9.45%) | 9/127 (7.09%) |

These are **historical observations, not a controlled causal comparison**.
ReAct and the branching methods use different commits; the runs also predate
the October 1 finalizer/reward fixes. GAIA uses a local corpus rather than the
official tool setting. An artifact audit does not independently rejudge answers.

The saved **SWE-bench Verified** ARM pilot evaluated only `sympy__sympy-20590`:
FoldAgent produced an empty patch and resolved **0/1**. Prediction hashes,
patch bytes and grading metadata reconcile locally. The dataset's 500-task
catalog size is not the evaluated count, and `official_x86_result=false`.
See the [ARM evidence section](results/audited/README.md#swe-bench-verified-arm-compatibility-pilot).

## Current Vista RL profile

From a Vista login node, after the environment/preflight in the
[RL guide](docs/bcp_qwen35_9b_rl.md):

```bash
cd /work/09281/chc_1996/vista/context-graph
git pull --ff-only origin master
mkdir -p logs
sbatch scripts/train_bcp_qwen35_9b_contextgraph_32k_4x4_5node.sbatch
```

This requests five nodes: one retrieval node and four trainers. Configuration:
32K context, four prompts with four rollouts each, PPO minibatch setting four,
50 updates, 48-hour allocation. Branches can produce additional training
trajectories; 16 main rollouts does not imply 16 independent fixed-cost requests.
Pre-training validation can take substantial time before the first update.
Submission, dependency preflight and rollout progress do not prove a completed
optimizer step. Preserve the run's commit and do not update its shared checkout
while it is running.

## AgentFold-style zero-shot baseline

The independent `agentfold` method implements model-written suffix compression:
the latest step can be condensed, or a contiguous suffix of earlier summaries
and new evidence can be replaced by one summary. It is different from the existing
`foldagent` branch/return executor. No training or change to other methods is required.

This is a **zero-shot adaptation**, not a reproduction of the trained AgentFold
checkpoint. Its mechanism follows the [official inference implementation](https://github.com/Alibaba-NLP/DeepResearch/blob/f72f75d8c3eb842f2bbbab096a12206ff66e270f/WebAgent/AgentFold/infer.py),
using our existing XML search/open_page/finish tools, local corpus, backbone and
judge. Current support is BC-P and the local, text-only GAIA comparison; the latter
is not official GAIA evaluation. Other simulator/code environments are not wired.

On Vista, submit the saved four-node, 24-hour, eight-task smoke script:

```bash
sbatch scripts/eval_bcp_agentfold_qwen35_9b_4node.sbatch
```

Inside an existing four-node allocation, run
`bash scripts/eval_bcp_qwen35_9b_4node_idev.sh agentfold` instead. Set `SAMPLES=-1`
for a full evaluation. The Python entry point accepts `--method agentfold`;
the GAIA evaluator dispatches `--workflow search_agentfold`.

Defaults retain the shared 32K working window, 100 model turns and 24K response
allowance. The allowance charges cumulative generated token IDs (including
compression/retries) and tokenized observation/format-feedback text; folding
does not refund it. This explicit adaptation budget does not reproduce the
upstream 100K context/500-step setup, and chat-wrapper accounting can differ from
the older executors. Final-answer reservation may truncate observations, with a
visible marker and counters. Invalid folding consumes a model turn but cannot
execute a tool. The original question is never truncated. Raw observations,
exact model inputs/outputs, fold errors and final working history are retained in
trajectory artifacts; manifests identify the adaptation and pinned source.
CPU tests validate execution contracts, not zero-shot model performance.

## SUPO-style zero-shot baseline

The independent `supo` / `search_supo` executor reimplements the rollout in
[SUPO Algorithm 2](https://arxiv.org/abs/2510.06727v1). After a tool call crosses
the context threshold, its action/observation pair is excluded from working
history. The same model summarizes the preceding context, and execution resumes
from the original task plus that summary. The tool has already run: its effects
are not undone. Raw results and discarded rounds remain in audit artifacts.

This is a **paper-based zero-shot rollout adaptation**, not SUPO joint RL training
or evaluation of an official SUPO checkpoint. Training calls are rejected. It
currently supports BC-P and our local text-only GAIA setup. It does not change
the existing ReAct, FoldAgent, AgentFold or ContextGraph protocols.

For an existing four-node Vista idev allocation:

```bash
BENCHMARK=bcp SAMPLES=8 bash scripts/eval_bcp_qwen35_9b_4node_idev.sh supo
```

Alternatively, `sbatch scripts/eval_bcp_supo_qwen35_9b_4node.sbatch` requests four
nodes for 24 hours. Both default to Qwen3.5-9B and an eight-task smoke. Threshold
`SUPO_CONTEXT_THRESHOLD=16384`, `SUPO_MAX_SUMMARIES=2` and
`SUPO_SUMMARY_MAX_TOKENS=1024` are configurable and saved in the manifest. The
Python equivalents are `--supo-context-threshold`, `--supo-max-summaries`, and
`--supo-summary-max-tokens`.

The adaptation retains the 32K working window, 100 model requests (including
summaries and invalid calls) and 24K cumulative response allowance. Generated
IDs and visible observation/instruction text consume the allowance; a summary
never resets it. Discarded observations are not shown to the model and are
reported separately. This budget and the reserved final-answer phase are our
evaluation controls, not the paper's original long-horizon training protocol.
Invalid summaries stop with `invalid_summary`; exhausted summary count stops
with `summary_limit`. Neither means success. Inspect `summary_restarts` to verify
that a smoke actually exercised compression. GPU performance remains unverified.

## Runtime and code layout

| Directory | Responsibility |
| --- | --- |
| `agents/` | Executors, prompts, graph memory, controller and finalization |
| `envs/` | Search clients/server, benchmark tools, containers and task rewards |
| `scripts/` | Data preparation, launchers, evaluation and evidence audits |
| `verl/` | Vendored distributed training implementation and compatibility changes |
| `tests/` | CPU regressions, protocol checks and optional runtime integrations |
| `docs/` | Architecture, protocols, cluster instructions and review findings |
| `results/` | Published evidence summaries and sanitized audited records |

Global and isolated ContextGraph are distinct variants. The isolated executor's
`legacy`, `repaired` and `foldagent` memory modes are experimental controls.
Training keeps generated history immutable; graph pruning is not a guarantee
that the policy's entire training context shrinks. Branch-return summaries and
graph merge summaries remain supported. Opt-in `enable_summary=True` restarts
the working context using a bounded summary, keeps the original total token
budget, and exports separate training segments with their original log-probs.
Existing launch profiles keep this option disabled by default.

## Validation and reproducibility

Install the dependencies appropriate to the selected workflow. The full test
suite requires training dependencies such as Ray, cloudpickle and TensorDict;
GPU/runtime checks additionally require the actual Vista environment.

```bash
python -m pytest -q tests
python scripts/export_audited_results.py --source outputs --destination results/audited --arm-source logs/swe-foldagent-1035199-hgEsOA
```

The export command requires the original synced artifacts. Raw trajectories,
model weights, credentials and generated logs are not published. See the
[review report](docs/project_review_20261001.md) for exact validation boundaries
and the [architecture](docs/contextgraph_architecture.md) for research design.
