# Running on TACC Vista (GH200 / aarch64)

Working notes from a run attempt on Apr 18-19 2026. 4B smoke passed; 30B smoke still fails at vLLM init.

## Environment

- conda env `cxtgraph` (py 3.10): torch 2.7.1+cu128, **vllm 0.10.1 (source-built)**, flash-attn 2.8.3 (source-built), transformers 4.57.6, compressed_tensors 0.14.0 (copied from the qerl repo env — see below)
- NOT torch 2.10 / vllm 0.19.x — the vendored `verl` imports `vllm.lora.models.LoRAModel` which only exists in vllm 0.10.x; the later versions renamed it to `vllm.lora.lora_model`
- For `test_alfworld_30b`, also requires `textworld` 1.7.0 and `alfworld` + `fast-downward-textworld` 20.6.4 — upstream textworld's `setup.sh` fails on aarch64 (Inform7 has no aarch64 binary); stub `setup.sh` with `exit 0` before `pip install .`; for `fast-downward-textworld` source-build via `pip install --no-build-isolation fast-downward-textworld`

## Missing config file (IMPORTANT)

`verl/trainer/config/data/legacy_data.yaml` is referenced by `ppo_trainer.yaml` defaults but the file was not present in this repo. I reconstructed it from the fields in `_generated_ppo_trainer.yaml`. Without it every run fails with `Could not find 'data/legacy_data'`.

## Vista-specific gotchas (saved me hours when I figured them out)

1. `~/.bashrc` prepends `~/.local/bin` which shadows the conda env's `pip` / `ray` binaries. After `conda activate`, always re-export: `export PATH=$CONDA_PREFIX/bin:$PATH; hash -r`. Also set `PYTHONNOUSERSITE=1` before anything Python.
2. NVIDIA HPC SDK's `nvc++` is in `PATH` before `g++` and does not accept `-march=armv8.2-a+...` GCC flags. For any vllm/flash-attn source build, export `CC=gcc CXX=g++ CUDAHOSTCXX=g++ CMAKE_CUDA_HOST_COMPILER=g++` and put `/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/bin` at the front of `PATH`.
3. On multi-node runs, Triton's kernel cache races when multiple ranks write to the same path on `$WORK` (NFS). Set `export TRITON_CACHE_DIR=/tmp/triton_cache_$$` inside each rank to use node-local `/tmp`.
4. Ray's default object-store reservation (~30% of node RAM = ~60 GB on a 212 GB Vista node) eats into the available CPU RAM. Add `--object-store-memory 5000000000` to `ray start` on both head and worker nodes, and optionally `export RAY_memory_usage_threshold=0.99` if peaks briefly exceed default 0.95.
5. Allocation on Vista is `AST24021` (not `ASC24078`). Partition `gh-dev` is capped at 2h walltime but usually has idle nodes; `gh` allows up to 2 days but long queue.

## Smoke-test status

- 4B `scripts/test_alfworld_4b_1node_1h.sh`: **PASS** — 3/3 FoldGRPO steps, rewards 0 → 0.06 → 0.31 across the 3 steps after fixing the fast-downward 32-bit→64-bit issue.
- 30B `scripts/test_alfworld_30b_2node_1h.sh`: **NOT PASSING** on 2 or 8 nodes. Failure chain documented below.

## 30B failure root cause (best guess; please verify)

FSDP FULL_SHARD is not actually happening across the 8 ranks. Each GPU still holds a full ~60 GB copy of Qwen3-30B bf16 post-load, so vLLM errors at init with "Free memory on device (8.39/95.0 GiB) < desired gpu_memory_utilization (0.35, 33.25 GiB)" — the symptom is exactly one un-sharded model per GPU.

I set the usual FSDP keys:

```
actor_rollout_ref.actor.strategy=fsdp
actor_rollout_ref.actor.fsdp_config.fsdp_size=-1
actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True
actor_rollout_ref.ref.fsdp_config.fsdp_size=-1
actor_rollout_ref.ref.fsdp_config.reshard_after_forward=True
```

Codex suggested the root cause is that each Ray WorkerDict sees its own `WORLD_SIZE=1` torch.distributed env, so FSDP has nothing to shard across even though `fsdp_size=-1` would otherwise mean "full mesh". Recommended check:

```python
# in verl/workers/fsdp_workers.py init_model(), after mesh creation:
assert torch.distributed.get_world_size() == int(os.environ["WORLD_SIZE"])
logger.warning("FSDP rank=%s world_size=%s mesh=%s", self.rank, world_size, self.device_mesh)
```

If `get_world_size()` prints 1 per rank, that's the bug — the 8 ranks never joined a single process group. Upstream verl's `RayWorkerGroup` (`verl/single_controller/ray/base.py` `_init_with_resource_pool`) is supposed to set `WORLD_SIZE=resource_pool.world_size`; may be broken or configured differently in this vendored copy.

## vLLM on 8 x 1-GPU nodes

- `tensor_model_parallel_size=8` across 8 one-GPU nodes: allowed by vLLM, but slow/fragile over IB.
- `pipeline_model_parallel_size=8`: **not implemented** in this vendored verl — raises `NotImplementedError: Current rollout self.name='vllm' not implemented pipeline_model_parallel_size > 1 yet.` So PP isn't an option until verl wires it through.

## Patches in this branch

See commit message on HEAD. Both scripts carry my `yw23374` / `AST24021` paths — revert to your own before re-running. The **non-path** fixes worth keeping on master:
- `--per-device-train-batch-size` / `num_generations` compatibility
- `--object-store-memory` cap on `ray start`
- `TRITON_CACHE_DIR=/tmp/$$` export
- `PATH=$CONDA_PREFIX/bin:$PATH` prepended inside every `srun bash -c "..."` subshell
- `actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1` (avoids the ValueError in trl 0.21)
- `verl/trainer/config/data/legacy_data.yaml` (this commit adds the missing file)
