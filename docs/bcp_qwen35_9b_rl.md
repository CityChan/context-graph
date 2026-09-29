# Qwen3.5-9B BC-P RL, 50 steps

Submit both independent jobs from the Vista login node:

```bash
cd /work/09281/chc_1996/vista/context-graph && git pull --ff-only origin master && bash scripts/submit_train_bcp_qwen35_9b_50step.sh
```

The submitter accepts `contextgraph` or `foldagent` to submit only one method.
Each job requests five GH200 nodes for 48 hours with account AST24021: one
retriever and four FSDP trainer ranks (ten nodes if both run simultaneously).
The one-hour ReAct evaluation limit does not apply to these training jobs.

## Configuration

| Setting | Value |
|---|---|
| Base checkpoint | Qwen/Qwen3.5-9B, revision c202236235762e1c871ad0ccb60c8ee5ba337b9a |
| Checkpoint location | `$SCRATCH/hf_cache/hub/models--Qwen--Qwen3.5-9B/snapshots/REVISION` |
| Data | `data/bc_train.parquet`; validation `data/bc_test.parquet` |
| Updates | 50, full parameter FSDP BF16, gradient checkpointing and CPU offload |
| Batch / samples per prompt | 32 / 8 |
| PPO minibatch setting | 32 per rank in this fork's padding convention |
| Context | 8192 prompt + 24576 response = 32768 |
| Optimizer | LR 1e-6, weight decay 0.1, grad clip 0.5; no KL loss |
| PPO clipping | low 0.2, high 0.28 |
| Rollout / validation | temperature 1 / greedy, validation n=1 |
| Validation / save | Before training and every 10 steps / every 5 steps |
| Branch budget | 10 branches; 100 turns; final-answer reserve 1024 |
| Search limits | top-k 5, snippet 128 words/2000 chars, page 4096 words/48000 chars |
| Seed | data seed 42 |
| Actor forward | native padded HF SDPA, no remove-padding/fused patch, SP=1 |
| Rollout | eager vLLM, language-model-only |

ContextGraph uses the isolated graph, repaired memory and structured controller,
matching the method used in the recent 9B evaluation; StructMem is off. No new
branch-forcing prompt is introduced. FoldAgent retains its existing prompt.

These are the repository's **local FoldGRPO recipes**, not an upstream-paper
reproduction or a reward-matched memory-only ablation. Method-specific reward
defaults are retained: FoldAgent `[flat,scope]` with `paper_signed` process
credit; ContextGraph `[flat,scope,graph]` with `relative_extrema`, graph shaping
and controller penalties. Both runs share the base model and optimization
settings but differ in these rewards as well as prompts/memory mechanics.

The existing 8B-named base scripts are reused as generic launchers. They now
accept a training environment and trailing Hydra overrides; their default 8B
behavior remains unchanged. The wrapper pins the 9B snapshot, fresh output and
checkpoint directories, and disables checkpoint resume. HF downloads are not
performed. Saved training checkpoints go to `$SCRATCH/context-graph-ckpts/`.

## Runtime checks and validation boundary

Trainer default: `TRAIN_CONDA_ENV=deepseek_v4`; retriever: `SEARCH_CONDA_ENV=cxtgraph`.
The former was used for 9B serving; **serving success does not demonstrate VERL
training compatibility**. The batch first checks complete checkpoint shards,
model/tokenizer classes, imports of both agent training loops, FSDP and vLLM
rollout, controller sampling API, and the observation tokenizer regression.
The tokenizer regression uses Python's built-in `unittest`; `pytest` is not
required in the training environment. It loads the pinned local tokenizer and
checks both short branch evidence and long search observations without weights.
`AgentContext` and the RL prompt-length filter explicitly request token lists:
Transformers 5.15.1 defaults to `BatchEncoding`, whose length/slicing do not
represent token IDs. Job `1033347` exposed this at the tokenizer preflight.
The fix was reproduced and verified with the pinned 9B tokenizer revision under
both Transformers 4.57.6 and 5.15.1, including exact suffix matching, preservation
of generated IDs/log probabilities/masks, observation replacement and rollback.
Missing dependencies or incompatible APIs stop the job before retrieval startup.
It never installs or upgrades packages in the shared environment.

The vLLM adapter supports the legacy GuidedDecodingParams API and the modern
StructuredOutputsParams replacement, documented in the
[vLLM structured outputs guide](https://github.com/vllm-project/vllm/blob/main/docs/features/structured_outputs.md).
The Qwen3.5 path disables this fork's untested sequence-packing patches.

Local syntax/unit tests do not establish CUDA backward, distributed weight
transfer, memory fit, or 50-step convergence on Vista. Those remain runtime
validation items; a preflight pass is not a successful training step.

Local validation: Bash syntax checks and 67 targeted tests passed (sampling
adapters, checkpoint validation, mocked Slurm submissions and existing training
wiring). One pre-existing ALFWorld source-string assertion in
`test_verl_controller_eval_wiring.py` fails against the unchanged graph agent;
it was excluded from the passing targeted run, not repaired by this change.

To submit a dependency-only job (one node, ten minutes, no optimization):

```bash
bash scripts/submit_train_bcp_qwen35_9b_50step.sh preflight
```

This single job checks both agent entry points in separate Python processes.
Failed checks include full tracebacks in `preflight.json`. It does not start
the retriever or test a distributed training step.

Jobs `1031462` (ContextGraph) and `1031463` (FoldAgent), launched from `66099ff`,
stopped in this dependency check: direct import of `scripts.train_fold` caused
a circular import, and the old LoRA helper imported `vllm.lora.models`, absent
in the installed vLLM 0.27.1. Their `.err` files were empty because errors were
redirected into `.out` and `suite.log`. No training started.

The agent package now defers access to training classes during registration.
Full-parameter BF16 startup no longer eagerly loads the version-specific LoRA
patch or FP8 MoE classes. This does not establish compatibility for LoRA or FP8
training on vLLM 0.27.1. Re-run the short preflight before submitting both
five-node training jobs.

Startup-fix validation: 46 targeted tests passed, including fresh import-order
regressions, optional LoRA/FP8 import boundaries, mocked Slurm submission and
failure-log routing; both changed Bash launchers passed syntax checks.

## Artifacts

### Four-node idev training smoke

After preflight passes, use the existing four-node allocation:

```bash
cd /work/09281/chc_1996/vista/context-graph && git pull --ff-only origin master && bash scripts/smoke_bcp_qwen35_9b_4node_idev.sh contextgraph
```

Use `foldagent` for the other method, or `both` to run them sequentially and stop
on the first failure. Do not run both concurrently in the same allocation.
One node serves retrieval and three train the same pinned full-parameter BF16
9B model. Smoke overrides: one update, 3 prompts x 2 samples, PPO mini-batch 3
(2 trajectories per rank before trajectory padding), 8192+4096 token budget,
4 turns, 1 branch, 512 tokens per turn, 600-second session timeout. It disables
pre-training/periodic validation and W&B, and saves at step 1. This checks the
initial weight transfer, rollout, backward/update and checkpoint-writing path;
it does not verify a second rollout with updated weights or 32K training fit.
The allocation's remaining walltime still applies; runtime is not guaranteed.

Artifacts go to `outputs/smoke_qwen35_9b_bcp_METHOD_JOBID_TIMESTAMP/` and a fresh
scratch checkpoint directory. `BCP_RL_SMOKE_COMPLETE` requires trainer exit 0,
the step-1 marker, and nonempty model/optimizer/extra-state shards for all three
ranks. This checks file presence, not checkpoint reload correctness or nonzero
learning signal. No benchmark accuracy or convergence claim follows from it.
The ordinary five-node, 50-step submission keeps its original configuration.

- Slurm: `logs/bcp-9b-rl50.METHOD-JOB-NAME.JOBID.out` and `.err` (the exact job
  name is `bcp-9b-contextgraph-50` or `bcp-9b-foldagent-50`).
- Suite: `outputs/train_qwen35_9b_bcp_METHOD_JOBID_TIMESTAMP/`, including
  `suite.log`, `commit.txt`, `preflight.json`, `data.sha256`, `overrides.txt`.
  Files after the failed stage may not exist. Failures after suite setup now
  also write `failure.log` and a stage/exit-code summary to Slurm `.err`.
- Search/Ray logs: `logs/`, filenames include job ID and run tag.
- Checkpoints: the suite prints the exact scratch path. Confirm a saved
  `global_step_50` and `latest_checkpointed_iteration.txt`; a submission or
  dependency preflight alone is not completion.

Do not update the shared checkout while either training job is running.
