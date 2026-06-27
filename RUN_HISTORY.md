# BrowseComp-Plus 30B RL — Run History

> Qwen3-30B-A3B-Instruct-2507 · FoldGRPO · Vista GH200 · wandb: [huancheng/context-graph](https://wandb.ai/huancheng/context-graph) (user: lingchensanwen)

## All Runs (BrowseComp 30B only)

### Early Runs (06-11 ~ 06-15)

| Date | wandb Run Name | KL | LR | clip_high | Quant | gpu_mem | Steps | State | wandb |
|------|----------------|-----|-----|-----------|-------|---------|-------|-------|-------|
| 06-11 | train_ctxgraph_bc_30b_instruct_taskreward_32n_8h | 0 | 2e-6 | 0.28 | FP8 | 0.55 | 50 | crashed | [link](https://wandb.ai/huancheng/context-graph/runs/y771c5a6) |
| 06-13 | debug_ctxgraph_bc_30b_instruct_v3localjudge_2h | 0.0005 | 2e-6 | 0.2 | FP8 | 0.55 | 10 | crashed | [link](https://wandb.ai/huancheng/context-graph/runs/7nmcux2p) |
| 06-14 | train_ctxgraph_bc_30b_instruct_qwen32judge_9n_8h | 0.0005 | 2e-6 | 0.2 | FP8 | 0.55 | 50 | crashed | [link](https://wandb.ai/huancheng/context-graph/runs/cx7sxtj3) |
| 06-14 | train_ctxgraph_bc_30b_instruct_qwen32judge_32n_8h | 0 | 2e-6 | 0.28 | FP8 | 0.55 | 50 | crashed | [link](https://wandb.ai/huancheng/context-graph/runs/z0zq7dev) |

### Tuning Runs (06-19 ~ 06-22)

| Date | wandb Run Name | KL | LR | clip_high | Quant | gpu_mem | entropy | Steps | State | wandb |
|------|----------------|-----|-----|-----------|-------|---------|---------|-------|-------|-------|
| 06-19 | train_..._qwen32judge_32n_8h_20260619 | 0.0005 | 2e-6 | 0.2 | FP8 | 0.55 | 0 | 50 | crashed | [link](https://wandb.ai/huancheng/context-graph/runs/gvh7ohko) |
| 06-20 | train_..._qwen32judge_32n_8h_20260620 | 0.002 | 2e-6 | 0.2 | FP8 | 0.55 | 0 | 50 | crashed | [link](https://wandb.ai/huancheng/context-graph/runs/9yif5382) |
| 06-21 | train_..._qwen32judge_32n_8h_20260621_115805 | 0.001 | 1e-6 | 0.2 | FP8 | 0.55 | 0 | 50 | finished | [link](https://wandb.ai/huancheng/context-graph/runs/9917k53y) |
| 06-22 | train_..._qwen32judge_32n_8h_20260622_095905 (a) | 0.0005 | 2e-6 | 0.2 | FP8 | 0.45 | 0 | 50 | crashed | [link](https://wandb.ai/huancheng/context-graph/runs/ry3lshk8) |
| 06-22 | train_..._qwen32judge_32n_8h_20260622_095905 (b) | 0.002 | 1e-6 | 0.2 | FP8 | 0.45 | 0 | 50 | crashed | [link](https://wandb.ai/huancheng/context-graph/runs/7s9ilk2c) |

### Current Experiment Runs (06-22 ~ now)

| Date | Slurm Job | wandb Run Name | KL | LR | clip_high | Quant | gpu_mem | entropy | Steps | Val Best | State | wandb |
|------|-----------|----------------|-----|-----|-----------|-------|---------|---------|-------|----------|-------|-------|
| 06-22 | 780702 | train_..._20260622_171454 | 0.002 | 1e-6 | 0.2 | BF16 | 0.75 | 0 | 11 | 0.573→0.500↓ | crashed | [link](https://wandb.ai/huancheng/context-graph/runs/w490iomr) |
| 06-22 | 781336 | train_..._20260622_235735 | 0.015 | 1e-6 | 0.2 | BF16 | 0.75 | 0 | 19 | 0.533→0.573→0.533 | crashed | [link](https://wandb.ai/huancheng/context-graph/runs/kk3ue9nu) |
| 06-23 | 782501 | train_..._20260623_144722 | 0.015 | 3e-7 | 0.15 | BF16 | 0.75 | 0 | 0 | — | crashed | [link](https://wandb.ai/huancheng/context-graph/runs/3wv3rf4z) |
| 06-23 | 783121 | debug_..._20260623_181815 | 0.0005 | 2e-6 | 0.2 | FP8 | 0.55 | 0 | 1 | — | crashed | [link](https://wandb.ai/huancheng/context-graph/runs/b1o9q1s6) |
| 06-24 | 783411 | debug_..._20260624_011124 | 0.0005 | 2e-6 | 0.2 | FP8 | 0.55 | 0 | 1 | — | crashed | [link](https://wandb.ai/huancheng/context-graph/runs/vzs1omvf) |
| 06-24 | 784989 | train_..._20260624_170248 | 0.0005 | 2e-6 | 0.2 | BF16 | 0.75 | 0 | 6 | 0.567→0.527↓ | crashed | [link](https://wandb.ai/huancheng/context-graph/runs/4ea17lur) |
| 06-25 | 786956 | train_..._20260625_132753 | 0.0005 | 2e-6 | 0.2 | BF16 | 0.75 | 0 | 0 | val=0.580 | finished | [link](https://wandb.ai/huancheng/context-graph/runs/5z2d92q2) |
| 06-25 | 787422 | train_..._20260625_201028 | 0.015 | 1e-6 | 0.2 | FP8 | 0.55 | **0.01** | 0 | val=0.460 | finished | [link](https://wandb.ai/huancheng/context-graph/runs/t0tv1co7) |
| 06-26 | **788013** | **train_..._20260626_105758** | **0.015** | **1e-6** | 0.2 | **FP8** | **0.55** | 0 | **15+** | 0.513→0.520→0.500 | **🟢 running** | [link](https://wandb.ai/huancheng/context-graph/runs/lfjmk7f1) |

### Shared Params (all runs)

| Param | Value |
|-------|-------|
| batch_size | 14 |
| rollout_n | 8 |
| response_length | 32768 |
| model | Qwen3-30B-A3B-Instruct-2507 |
| framework | veRL / FoldGRPO |
| cluster | Vista (TACC GH200 96GB) |

## Key Findings

- **KL penalty 无效**: ppo_kl ≈ 0，所有 KL 值 (0 / 0.0005 / 0.001 / 0.002 / 0.015) 都一样
- **BF16 在 28 节点不可用**: 786502/786956 都 crash，FP8+0.55 稳定 (788013)
- **entropy 计算放不进 GPU**: 787422 加 entropy_coeff=0.01 → actor update OOM
- **无 entropy bonus 时 entropy 必然振荡**: 所有 run 都 collapse ↔ recover
- **val 从未持续改善**: 需要先解决 entropy regularization
- **gpu_mem=0.45 也 crash 过**: 早期 runs (ry3lshk8, 7s9ilk2c) 在 0.45 就挂了

*Updated: 2026-06-26*
