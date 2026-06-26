# BrowseComp-Plus 30B RL — Run History

> Qwen3-30B-A3B-Instruct-2507 · FoldGRPO · Vista GH200 · Baseline val/task_reward ~0.533

## All Runs

| Date | Job | Type | Quant | gpu_mem | KL | LR | entropy_coeff | Steps | Val Best | Entropy | 结局 |
|------|-----|------|-------|---------|-----|-----|---------------|-------|----------|---------|------|
| 06-21 | 780702 | prod 28n | BF16 | 0.75 | 0.002 | 1e-6 | 0 | 11 | 0.573→0.500↓ | 永久 collapse (step 6+) | node core dump |
| 06-22 | 781336 | prod 28n | BF16 | 0.75 | 0.015 | 1e-6 | 0 | 19 | 0.533→0.573→0.533 | 振荡 0.01↔0.37 | node core dump step 19 |
| 06-23 | 782501 | prod 28n | BF16 | 0.75 | 0.015 | 3e-7 | 0 | 0 | — | — | cancelled (LR analysis) |
| 06-23 | 783121 | canary 7n | FP8 | 0.55 | 0.0005 | 2e-6 | 0 | 1 | — | 0.039 (near collapse) | failure_shaping=0.4 太高 |
| 06-23 | 783411 | canary 7n | FP8 | 0.55 | 0.0005 | 2e-6 | 0 | 1 | — | **0.403 ✅** | failure_shaping=0.2 唯一健康 |
| 06-24 | 784989 | prod 28n | FP8 | 0.75† | 0.0005 | 2e-6 | 0 | 6 | 0.567→0.527↓ | 振荡 0.01↔0.30 | CPU OOM (config mismatch) |
| 06-25 | 786502 | prod 28n | BF16 | 0.55 | 0.0005 | 2e-6 | 0 | 0 | — | — | 即时 OOM (BF16+0.55) |
| 06-25 | 786956 | prod 28n | BF16 | 0.75 | 0.0005 | 2e-6 | 0 | 0 | val=0.580 | — | EngineCore crash 49min |
| 06-25 | 787422 | prod 28n | FP8 | 0.55 | 0.015 | 1e-6 | **0.01** | 0 | val=0.460 | — | entropy 计算 OOM |
| 06-26 | **788013** | **prod 28n** | **FP8** | **0.55** | **0.015** | **1e-6** | 0 | **14+** | 0.513→0.520→0.500 | 振荡 0.01↔0.43 | **🟢 RUNNING** |

> † 784989 有 3 个 config override 跟 canary 不一致
> 787350/787391 cancelled，未列入

## Key Findings

- **KL penalty 无效**: ppo_kl ≈ 0，所有 KL 值都一样
- **BF16 在 28 节点不可用**: 786502/786956 都 crash，FP8+0.55 稳定 (788013)
- **entropy 计算放不进 GPU**: 787422 加 entropy_coeff=0.01 → OOM
- **无 entropy bonus 时 entropy 必然振荡**: 所有 run 都 collapse ↔ recover
- **val 从未持续改善**: 需要先解决 entropy regularization

*Updated: 2026-06-26*
