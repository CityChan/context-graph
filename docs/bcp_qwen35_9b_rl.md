# Qwen3.5-9B BC-P RL, 50 steps

## Five-node ContextGraph 32K, batch 4 x rollout 4

Submit from a Vista login node:

```bash
cd /work/09281/chc_1996/vista/context-graph && git pull --ff-only origin master && mkdir -p logs && sbatch scripts/train_bcp_qwen35_9b_contextgraph_32k_4x4_5node.sbatch
```

This profile uses one search node and four trainer nodes, with 4 prompts x 4
rollouts = 16 global sequence slots and PPO minibatch setting 4 (four slots per
trainer rank after rollout expansion). It retains the shared 32K context,
50 steps, validation, and checkpoint schedule. Slurm requests 48 hours; logs
are `logs/bcp-9b-cg-4x4.JOBID.out` and `.err`. This is a smaller training batch
than the prior 4-node FoldAgent 9 x 4 run, so it is not a batch-matched comparison.

**Validation boundary:** launcher/profile tests establish argument wiring only.
The 32K, 4 x 4, five-node topology has not yet demonstrated a completed GPU
optimizer step or checkpoint reload in the locally available evidence. Run the
short smoke below first. Pre-training validation and multi-turn rollouts can
leave the update progress bar at zero while generation is active.

The October 1 review fixes final-answer acceptance, judge cleanup and search
result delivery; old scores retain the earlier protocol. `enable_summary=True`
is rejected because legacy session restart did not switch the active trajectory.
See the [review and open decisions](project_review_20261001.md), including the
success-gated graph cost and inherited branch-budget semantics.

## Four active nodes in idev (optional 9 x 4 profile)

To use the entire four-node allocation, the first node serves retrieval and the
remaining three train. Batch 9 x rollout 4 gives 36 main trajectories and 36
PPO sequence slots, or 12 per trainer rank. This changes the batch from the
8 x 4 profile below; both methods still use 32K context and 50 steps.

Run one method at a time in the existing idev allocation:

```bash
cd /work/09281/chc_1996/vista/context-graph && bash scripts/train_bcp_qwen35_9b_foldagent_32k_9x4_4node_idev.sh
```

```bash
cd /work/09281/chc_1996/vista/context-graph && bash scripts/train_bcp_qwen35_9b_contextgraph_32k_9x4_4node_idev.sh
```

These entry points do not submit Slurm jobs or extend the idev walltime. The
existing 8 x 4 entry points remain available below. A 32K training step on
this three-rank topology has not yet been verified on Vista.

## FoldAgent in the existing four-node idev

Use the same 32K, 50-step optimization configuration and node layout as the
ContextGraph idev run:

```bash
cd /work/09281/chc_1996/vista/context-graph && git pull --ff-only origin master && bash scripts/train_bcp_qwen35_9b_foldagent_32k_4node_idev.sh
```

Qwen3.5-9B revision, data, 8192+24576 token budget, batch 8 x 4, PPO minibatch
8, microbatch 1, learning rate, clipping, validation and checkpoint frequency
all use the shared wrapper. First node serves retrieval; the next two train;
the fourth is untouched. FoldAgent retains its own agent and `[flat,scope]`
`paper_signed` process rewards, rather than ContextGraph's graph rewards.
This is a fresh base-model training run, not a resume from ContextGraph.

Do not start the two methods concurrently in the same allocation: they use the
same nodes, Ray cluster and search port. If ContextGraph is still running, wait
for it to finish before pulling or launching FoldAgent. The existing idev
walltime applies; this script does not submit a batch job or extend the allocation.
The 12K smoke passed previously; 32K training on two GPUs remains unverified.

Logs and `training-config.txt` are captured under
`outputs/train32k_qwen35_9b_bcp_foldagent_JOBID_TIMESTAMP/`; checkpoints use
the corresponding experiment directory under `$SCRATCH/context-graph-ckpts/`.

## ContextGraph in the existing four-node idev

Current ContextGraph run: **32K, batch 8 x 4**, 50 steps.
Run directly in the existing **four-node idev allocation**:

```bash
cd /work/09281/chc_1996/vista/context-graph && git pull --ff-only origin master && bash scripts/train_bcp_qwen35_9b_contextgraph_32k_4node_idev.sh
```

This uses the first allocated node for retrieval and the next two for FSDP
training. The fourth node is excluded from service launches and Ray cleanup.
The batch is 8 prompts x 4 rollouts, PPO minibatch configuration 8 and
microbatch 1. Two trainer ranks divide the main-rollout count (32) and global
PPO sequence-slot count (32) exactly; three trainer ranks would not. Per-rank
normalized PPO minibatch is 16 slots, accumulated one sequence at a time.
Both idev launchers select a `*_32k_small_batch` profile. This reduces the main
rollout count eightfold from the previous 32 x 8 setting; it is no longer the
paper batch configuration. Historical paper-batch profiles remain available.
The smaller batch still needs a complete 32K training-step validation; it does
not guarantee that rollout timeouts or memory pressure are eliminated.
The Slurm allocation variables remain unchanged.

The existing idev walltime applies; the launcher does not submit or extend a
job. It captures `suite.log` automatically and records active/unused nodes in
`training-config.txt`. Two training GPUs have more model/optimizer state per
GPU than the tested three-rank 12K smoke, so 32K fit and completion of 50 steps
within the remaining walltime are unverified.

For a separate five-node batch allocation, the optional submission is:

```bash
cd /work/09281/chc_1996/vista/context-graph && git pull --ff-only origin master && mkdir -p logs && sbatch scripts/train_bcp_qwen35_9b_contextgraph_32k.sbatch
```

Both entry points use 8192 prompt + 24576 response tokens, 32 prompts x 8 rollouts,
PPO minibatch configuration 128 and microbatch 1 per GPU. The batch entry requests five
nodes (one search + four trainers) for 48 hours. The PPO batching semantics
described below also apply here. This new 32K profile replaces the proposed
64K run; selecting the older generic submitter would retain PPO minibatch 32.

Artifacts: `outputs/train32k_qwen35_9b_bcp_contextgraph_JOBID_TIMESTAMP/`,
including `training-config.txt`. Slurm logs: `logs/bcp-9b-cg-32k.JOBID.out`
and `.err`. This is a launch configuration, not evidence of a completed run;
32K backward memory fit remains to be validated on Vista.

## Original 32K recipes (PPO minibatch configuration 32)

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
| PPO minibatch setting | 32 prompt units globally; expanded by rollout n before division across trainer ranks |
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

### Optional ContextGraph 64K with FoldAgent batch settings

Submit this separate 50-step profile from a Vista login node:

```bash
cd /work/09281/chc_1996/vista/context-graph && git pull --ff-only origin master && mkdir -p logs && sbatch scripts/train_bcp_qwen35_9b_contextgraph_64k.sbatch
```

The supplied *Scaling Long-Horizon LLM Agent via Context-Folding* PDF, page 5,
Section 5.2, reports rollout batch 32, group size 8, PPO batch 128 and 50 steps.
The [pinned official Qwen3-8B launcher](https://github.com/sunnweiwei/FoldAgent/blob/58a2d6964ecebe99940529eace50a0558901b8a5/scripts/train_bc_qwen3_8b.sh)
sets `data.train_batch_size=32`, `rollout.n=8`, and
`actor.ppo_mini_batch_size=128`; this profile aligns those configuration fields.

| Setting | 64K profile |
|---|---|
| Backbone | Pinned Qwen3.5-9B checkpoint above |
| Prompt / response / active context | 8192 / 57344 / 65536 |
| Training prompts / samples per prompt | 32 / 8 (256 main rollouts before branch expansion) |
| PPO minibatch configuration | 128 |
| PPO microbatch per GPU | 1, with gradient accumulation |
| Topology | 5 GH200 nodes: 1 search + 4 trainer ranks |
| Updates / requested walltime | 50 / 48 hours |

The existing FSDP implementation, also present in the pinned upstream checkout,
multiplies the PPO minibatch setting by rollout n before dividing over DP ranks:
`128 * 8 / 4 = 256` sequence slots per rank. In this repository the controller
pads the variable number of main/branch trajectories to a multiple of **1024**
global slots; dummy trajectories are masked. Thus `128` is a configuration
value, **not a promise of 128 real trajectories per optimizer update**. Padding
can increase compute and memory costs. We do not change this batching algorithm
or claim numerical equivalence to the paper's unspecified PPO batching units.

The current four-node smoke topology has three training ranks and cannot evenly
divide 256 main rollouts. The new entry point requires five nodes; it does not
silently change the batch size or reuse the reduced smoke settings.

Native padded HF SDPA, microbatch 1, gradient checkpointing and CPU offload are
retained. The [pinned Qwen3.5 configuration](https://huggingface.co/Qwen/Qwen3.5-9B/blob/c202236235762e1c871ad0ccb60c8ee5ba337b9a/config.json)
has 262144 native text positions, so the wrapper disables the old Qwen3-8B
launcher's automatic YaRN override. It does not alter the model's RoPE settings.

This is a batch-configuration-aligned ContextGraph experiment. The paper uses
Seed-OSS-36B, 32768 active context, FoldAgent rewards and asynchronous off-policy
rollouts; those are not reproduced by selecting this profile. Validation, search,
controller and reward settings retain the existing local recipe. The normal 32K
profile and 12K smoke retain their existing batch sizes.

Artifacts use `outputs/train64k_qwen35_9b_bcp_contextgraph_JOBID_TIMESTAMP/` and
the corresponding scratch checkpoint directory. `training-config.txt` records
the resolved context/batch settings and upstream reference, alongside the existing
commit/data/override files. Slurm logs are `logs/bcp-9b-cg-64k.JOBID.out` / `.err`.

Both methods completed the 12K one-step smoke in allocation 1033347. That proves
the tested rollout/update/checkpoint path; **64K memory fit, sustained training,
and a subsequent rollout using updated weights remain unverified**. Local tests
exercise parameter forwarding, topology rejection and absence of the YaRN
override with mocked site commands; they do not run GPU training.

### Dependency preflight

Before importing the training stack, the shared wrapper now checks the selected
logging backend in the actual training interpreter and saves
`logging-preflight.json`. It loads the same `$WORK/.wandb_env` as the base
launcher and verifies the callable W&B SDK API without creating a run or making
network requests. Console-only and smoke runs do not require W&B; explicitly
requiring W&B while disabling it or omitting its key fails early.

ContextGraph run `1033347/20260929_140954` failed after model/vLLM startup,
before initial validation or any training update, because the imported `wandb`
module lacked `init`. That traceback does not establish an OOM. Check/install
the SDK in `deepseek_v4`, the interpreter used by the launcher, even if the
interactive shell says `cxtgraph`. A missing SDK can leave the repository's
`wandb/` output directory importable as a namespace; confirm the import path
before attributing the failure to that cause. Do not remove the log directory.
For an absent SDK, install it explicitly on Vista, outside any active run:

```bash
/work/09281/chc_1996/vista/miniconda3/envs/deepseek_v4/bin/python -m pip install wandb
```

Then rerun the selected idev entry point. The preflight will stop immediately
with interpreter/import-path diagnostics if installation is still invalid.
Authentication and network availability are checked by actual W&B startup,
not this offline API probe. No automatic SDK installation or logger fallback
is performed by the launcher.

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

The first smoke in allocation `1033347` (`20260929_113214`) reached the three
trainer ranks and loaded actor weights, then failed before rollout at
`WorkerWrapperBase(vllm_config=...)`. vLLM 0.27.1 receives this configuration
through `init_worker`, and no longer exposes the old wrapper `execute_method`.
The adapter now selects constructor arguments by signature and preserves wrapper
device/cache hooks during direct RPC dispatch. It also supports the executor's
`sample_tokens(grammar_output, non_block=...)` interface. See the pinned
[worker source](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/v1/worker/worker_base.py)
and [executor source](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/v1/executor/abstract.py).
Preflight checks these worker/adapter signatures before retrieval startup.
Validation: 36 targeted tests, plus CPU execution of the upstream 0.11.0 and
0.27.1 wrapper constructors and initialization/device/cache methods with stub
workers. Actual GPU initialization, weight transfer and training still require
the next smoke run; this is not a full vLLM-version compatibility guarantee.

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
