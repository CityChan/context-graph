# Running on TACC Vista (GH200 / aarch64)

Status as of Apr 22, 2026.

## Working 30B setup on this branch

The working Vista path is based on Chen's Apr 19, 2026 commit `fa2ce1c`
(`enable fp8 rollout quantization on 30B ALFWorld scripts`).

Intended precision split:
- actor/ref: BF16 via FSDP
- rollout engine dtype: BF16
- rollout weight quantization: FP8

This branch now has a reproducible Vista smoke script for that setup:
- `scripts/test_alfworld_30b_8node_2h_chen_fp8.sh`

It expects an existing 8-node idev allocation and launches the trainer on the
allocation head node instead of the login node.

## What passed

The Chen-style FP8 smoke on 8 Vista GH200 nodes completed all 3 training steps.
The run finished with `SMOKE TEST PASSED` and reached:
- `Training Progress: 3/3`
- wandb run `7r00zzo9`
- rollout quantization `fp8`
- actor/ref `model_dtype=bfloat16`

The reward stayed zero in this smoke. That is a task-performance result, not an
FP8 infrastructure failure.

## Reproduction

From the login node, after obtaining an 8-node idev allocation:

```bash
cd /work/07144/yw23374/vista/context-graph
source /work/07144/yw23374/vista/miniconda3/etc/profile.d/conda.sh
conda activate cxtgraph
export IDEV_JOBID=<your_idev_jobid>
bash scripts/test_alfworld_30b_8node_2h_chen_fp8.sh
```

If you are already inside the allocation shell, `IDEV_JOBID` is not required:

```bash
cd /work/07144/yw23374/vista/context-graph
source /work/07144/yw23374/vista/miniconda3/etc/profile.d/conda.sh
conda activate cxtgraph
bash scripts/test_alfworld_30b_8node_2h_chen_fp8.sh
```

## Branch-local fixes needed on Vista

These are the changes that made Chen's setup work here.

1. `verl/workers/fsdp_workers.py`
   Strip `quantization_config` from actor/ref model config so HF does not
   meta-initialize parameters and break FSDP1 broadcast.

2. `verl/trainer/constants_ppo.py`
   Preserve CUDA/compiler/library env vars in Ray runtime env so vLLM child
   processes can resolve CUDA shared libraries on Vista.

3. `scripts/test_alfworld_30b_8node_2h_chen_fp8.sh`
   Add Vista-safe idev orchestration:
   - clean stale Ray daemons on allocated nodes
   - start Ray on compute nodes only
   - use BF16 base model with rollout `quantization=fp8`
   - set rollout TP=1 to match Chen's scripts
   - disable vLLM sleep mode / free-cache engine on Vista

## Recommended test ladder

Submit each via `sbatch` from a Vista login node, in order. Move up only
after the previous tier passes:

1. `scripts/test_alfworld_4b_1node_1h.sh`
   4B model, 1 node, 1 hour. Validates conda env, CUDA, vLLM, ALFWorld import,
   and the basic training loop on a single GH200.
2. `scripts/test_alfworld_30b_4node_1h_chen_fp8.sh`
   30B Chen FP8, 4 nodes, 1 hour. First multi-node IB collective test; cheaper
   than 8-node and uses the same proven config.
3. `scripts/test_alfworld_30b_8node_2h_chen_fp8.sh`
   30B Chen FP8, 8 nodes, 2 hours. Production scale validation; this is the
   smoke that previously passed (wandb run `7r00zzo9`).
4. `scripts/train_alfworld_fold_30b_16node_48h.sh` and
   `scripts/train_alfworld_ctxgraph_30b_16node_48h.sh`
   16-node, 48h production runs.

## Important caveats

- The smoke disables vLLM sleep mode and the free-cache engine:
  - `actor_rollout_ref.rollout.free_cache_engine=False`
  - `+actor_rollout_ref.rollout.engine_kwargs.vllm.enable_sleep_mode=False`
  This avoids the Vista `libnvrtc.so.12` / cuMem sleep-mode failure.
- The run may print vLLM event-loop errors during shutdown after training has
  already completed. Those appeared in teardown, after the smoke had passed.

## Repo scripts aligned with Chen

These production scripts now again carry rollout FP8, matching Chen's setup:
- `scripts/train_alfworld_fold_30b_16node_48h.sh`
- `scripts/train_alfworld_ctxgraph_30b_16node_48h.sh`

The key setting is:
- `+actor_rollout_ref.rollout.quantization=fp8`

## Environment reference

Validated stack on Vista:
- torch `2.7.1+cu128`
- vllm `0.10.1` (source-built)
- transformers `4.57.6`
- 8 x NVIDIA GH200 (1 GPU per node)
