# Qwen3-30B-A3B-Thinking-2507 FoldGRPO 8-node Smoke Test 修改日志

环境: TACC Vista, 8× NVIDIA GH200 (aarch64, 95 GiB HBM/node, 212 GB CPU/node, 1 GPU/node, InfiniBand)
Stack: torch 2.7.1+cu128, vllm 0.10.1 (源码编译), transformers 4.57.6, verl (cxtgraph vendored fork)

## v1–v15: 环境搭建 + 基础配置修复 (跨越两天)

早期版本主要解决的不是模型训练问题, 而是把环境跑起来:

| Version | 修改点 | 结果 |
|---|---|---|
| v1-v5 | 初始脚本路径, W&B 集成 (entity=huancheng), SLURM 账号 (chc_1996 → yw23374, account AST24021) | 环境初始化 |
| v6 | vllm fp8 符号重命名: `_swap_w13_to_w31` → `swap_w13_to_w31` 在 `verl/utils/vllm/vllm_fp8_utils.py` | fp8 import fix |
| v7 | vllm fp8 符号重命名: `is_blackwell_deep_gemm_used` → `is_blackwell_deep_gemm_e8m0_used` | fp8 import fix |
| v8 | 补齐缺失的 `verl/trainer/config/data/legacy_data.yaml` (从 `_generated_ppo_trainer.yaml` 重建) | Hydra config loads |
| v9 | 加 `actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1` 避免 ref 配置 ValueError | ref 配置通过 |
| v10-v12 | Ray CPU OOM: 加 `--object-store-memory 5000000000` 到 ray start, 加 `RAY_memory_usage_threshold=0.99`, `RAY_memory_monitor_refresh_ms=0` | Ray daemon 不再 OOM |
| v13 | NFS flock errors: 设 `TRITON_CACHE_DIR=/tmp/triton_cache_$$` + `VLLM_CACHE_ROOT=/tmp/vllm_cache_$$` 避 NFS 锁竞争 | triton 加载通过 |
| v14 | `HF_HUB_DISABLE_FILE_LOCKING=1` | HF cache 锁竞争缓解 |
| v15 | Hydra: `+actor_rollout_ref.rollout.quantization=fp8`, `gpu_memory_utilization=0.25` | fp8 rollout 启用 |

## v16-v21: 单节点/多节点过渡

| Version | 修改点 | 结果 |
|---|---|---|
| v16-v18 | 把 `trainer.n_gpus_per_node=1 trainer.nnodes=8` 改成 8 节点拓扑 (之前默认 1x8) | 多节点启动但 FSDP 没正确分片 |
| v19 | 加 `actor_rollout_ref.actor.fsdp_config.fsdp_size=-1`, `reshard_after_forward=True`, `param_offload=True`, `optimizer_offload=True` | FSDP 配置完整 |
| v20 | 试 2 节点 (`SLURM_JOB_NODELIST=c642-[001-002]`) — GPU OOM (tried 1.16 GiB, 1.89 free) | 2 节点不够 30B |
| v21 | 回 8 节点 + 加 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | 仍然 Duplicate GPU |

## v22-v24: NCCL "Duplicate GPU" 问题

| Version | 修改点 | 结果 |
|---|---|---|
| v22 | 原始 Ray+srun 配置 | `NCCL error ncclInvalidUsage: Duplicate GPU detected: rank 0 and rank 1 both on CUDA device 901000` |
| v23 | Codex 建议: 加 `export NCCL_HOSTID="${SLURMD_NODENAME:-$(hostname -s)}"` 在外层 + srun 块里 (转义: `\${SLURMD_NODENAME:-\$(hostname -s)}`) | 仍然 Duplicate GPU (stale Ray state) |
| v24 | 加 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | Duplicate GPU AGAIN (Ray 状态污染) |

## v25: NCCL 问题真正解决

| Version | 修改点 | 结果 |
|---|---|---|
| v25 | 全节点 Ray 清理 (`ray stop --force` + `pkill ray/train_fold/vllm` on all 8 nodes), 再用 `NCCL_HOSTID` 重启 | **NCCL Duplicate GPU 消失.** FSDP DIAGNOSTIC 打印 `world_size=8 env_WORLD_SIZE=8 mesh=DeviceMesh('cuda', [0..7])`. 模型 checkpoint shards 100% 加载. 但 **rank 0 (head node) CPU-OOM**: Qwen3-30B 以 fp32 (~120 GB/rank) 加载, head node 同时跑 Ray daemon + driver + rank 0 worker, 212 GB RAM 不够 |

## v26-v27: 模型 dtype 修复

| Version | 修改点 | 结果 |
|---|---|---|
| v26 | 加 `+actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16` | Hydra rejected `+` prefix: key already exists in default config |
| v27 | 改成 `model_dtype=bfloat16` (不带 `+`) | Actor bf16 加载 ✓, CPU OOM 解决. 但碰到 **vLLM init GPU OOM**: "Free memory (12.66/95.0 GiB) < gpu_memory_utilization (0.25, 23.75 GiB)". FSDP1 没有真正释放 82 GiB 的 GPU 缓存 (已知 verl bug #1149) |

## v28-v33: Hydra 前缀混乱 + vllm sleep mode 尝试

| Version | 修改点 | 结果 |
|---|---|---|
| v28 | 加 `+actor_rollout_ref.rollout.free_cache_engine=True`, `+actor_rollout_ref.rollout.engine_kwargs.vllm.enable_sleep_mode=True`, ref param_offload=True | `free_cache_engine` `+` prefix 被拒绝 (默认已是 True) |
| v29 | 去掉 `free_cache_engine` 的 `+` | 仍然被拒绝 (key-in-struct 错误) |
| v30 | 用 `++free_cache_engine=True` force override | 仍然被 Hydra 拒绝 |
| v31 | 完全删掉 `free_cache_engine` override (默认已是 True) | Stale Ray cluster 从之前运行占着 port 6379 → ConnectionError |
| v32 | 全 Ray 清理后再启, 保留 `+engine_kwargs.vllm.enable_sleep_mode=True` | **AssertionError**: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments` 与 vLLM memory pool 不兼容 |
| v33 | 注释掉 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | **静默挂起** — head node GPU 仍然 hold 36 GiB 的 FSDP1 残留 cache, 其他节点正常 |

## v34: 源代码补丁 + 诊断日志

| Version | 修改点 | 结果 |
|---|---|---|
| v34 | **源代码补丁** 到 `verl/workers/fsdp_workers.py` 第 831 行, 在 `_build_rollout()` 之前加 `torch.cuda.synchronize(); gc.collect(); torch.cuda.empty_cache()` + 诊断日志 `PRE-VLLM MEM: rank={} free={}GiB used={}GiB` | **重要诊断结果**: 7 个非 head 节点 free=93.9 GiB used=0.0 GiB (干净). **rank 0 (head) free=16.5 GiB** — nvidia-smi 显示 80 GiB 被其他 Python/Ray/vllm EngineCore 子进程占用. PyTorch `torch.cuda.memory_allocated()` 报 0 但 nvidia-smi 显示 80 GB — 说明是其他进程拿走的. vllm 仍然 OOM |

## v35: 全面切换到 FSDP2 (Codex 推荐)

| Version | 修改点 | 结果 |
|---|---|---|
| v35 | Codex 完整配方:<br>- `actor_rollout_ref.actor.strategy=fsdp2`<br>- `actor_rollout_ref.ref.strategy=fsdp2`<br>- 删除 `param_offload` / `optimizer_offload` (FSDP1-only)<br>- 加 `fsdp_config.offload_policy=True` (两边)<br>- `model_dtype=bfloat16` (两边)<br>- `reshard_after_forward=True` (两边)<br>- `rollout.gpu_memory_utilization 0.25 → 0.50`<br>- 加 `rollout.enforce_eager=True`<br>- 加 `rollout.load_format=dummy_dtensor`<br>- 加 `rollout.max_num_batched_tokens=2048`<br>- 加 `rollout.max_num_seqs=16`<br>- 把 `quantization=fp8` 移到 `engine_kwargs.vllm.quantization=fp8` | **GPU 内存大改善**: rank 0 head 从 16 → **51 GiB free**. 其他 7 个 rank 仍 93.9 GiB free. FSDP2 `offload_policy=True` 真的释放 GPU allocator cache. 但碰到 **新错误 (进步!)**: vLLM workers race on NFS HF cache downloading tokenizer_config.json blob, `FileNotFoundError: .incomplete` |

## v36: HF cache offline 模式 (未启动)

| Version | 修改点 | 结果 |
|---|---|---|
| v36 | 加 `export HF_HUB_OFFLINE=1` + `export TRANSFORMERS_OFFLINE=1` | **未启动** — 检查发现 Qwen3-30B 权重其实不在 `$HF_HOME` (只有 config + tokenizer, 52 MB). 60 GB 权重可能在 `$SCRATCH` 或其他位置. 强制 offline 可能会失败. 需要先确定权重的真实位置再决定 |

## 两个关键源代码修改 (持久)

1. `verl/workers/fsdp_workers.py` 第 831 行 (`_build_rollout` 之前):
```python
if self._is_rollout:
    import gc as _gc_ycheck
    torch.cuda.synchronize()
    _gc_ycheck.collect()
    torch.cuda.empty_cache()
    logger.warning(f'PRE-VLLM MEM: rank={self.rank} free={torch.cuda.mem_get_info()[0]/2**30:.1f}GiB used={torch.cuda.memory_allocated()/2**30:.1f}GiB')
    self._build_rollout(...)
```

2. `verl/workers/fsdp_workers.py` 第 167 行 (FSDP mesh 创建之后) — v25 以来一直保留:
```python
assert torch.distributed.get_world_size() == int(os.environ.get("WORLD_SIZE", "-1"))
logger.warning("FSDP DIAGNOSTIC rank=%s world_size=%s env_WORLD_SIZE=%s mesh=%s", ...)
```

## 真正的进步信号 (非自我折腾)

- **v25**: NCCL_HOSTID 解决 Duplicate GPU (持久收益, 所有后续版本都用)
- **v27**: bf16 model_dtype 解决 CPU OOM (持久收益)
- **v34**: 诊断证明 FSDP1 没有释放 GPU cache 在 rank 0 上 (定位问题)
- **v35**: FSDP2 + offload_policy=True 释放了 35 GiB GPU (16 → 51 GiB). **这是 35 版本以来第一次真正的内存改善**

## 自我反省

- v28-v31 纯粹是 Hydra `+` 前缀规则的混乱, 应该先读文档
- 应该在 v27 就直接切 FSDP2, 而不是继续折腾 FSDP1 offload flags
- v34 的源代码补丁虽然证明了问题, 但不是根治; FSDP2 才是
- 应该在依赖 offline 模式之前确认 HF cache 状态
