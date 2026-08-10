# BrowseComp-Plus Qwen3-8B 64K zero-shot evaluation

Frozen on 2026-08-09 from the completed full-test jobs. These are zero-shot
results and must not be overwritten by later guarded or trained evaluations.

## Shared evaluation configuration

- Model: `Qwen/Qwen3-8B`
- Test split: `data/bc_test.parquet`, all 150 samples
- Topology: four Vista GH nodes (one dedicated search node and three trainer nodes)
- Active context: 65,536 tokens (`prompt=8,192`, `response=57,344`)
- Long-context override: YaRN factor 2.0, original context 32,768
- Maximum sessions: 10
- Maximum turns: 100
- Session timeout: 600 seconds
- Validation: greedy, one trajectory per sample
- Repository head: `5b6a07b`

## Results

| Method | Slurm job | Correct | Task accuracy | Overlong | Average turns | Shaped reward |
|---|---:|---:|---:|---:|---:|---:|
| ReAct baseline | 899659 | 45/150 | 0.3000 | 40/150 (0.2667) | 16.2667 | 0.3000 |
| FoldAgent | 899660 | 14/150 | 0.0933 | 127/150 (0.8467) | 12.0200 | 0.0933 |
| ContextGraph | 899661 | 47/150 | 0.3133 | 11/150 (0.0733) | 24.7933 | 0.2992 |

For ContextGraph, `task_reward=0.3133` is the answer accuracy. Its
`val/reward=0.2992444` includes graph shaping and penalties and must not be
reported as accuracy.

## Additional diagnostics

- FoldAgent: `is_branch=0.8667`, `branch_success=0.0800`. The logs contain 196
  session-timeout events; the high overlong rate is primarily failure to return
  from branches before the common 600-second session deadline.
- ContextGraph: `is_branch=0.8733`, `branch_success=0.2800`, graph invalid-op
  rate `0.5556`. Despite frequent invalid graph operations, only 11 samples were
  overlong.
- ReAct baseline: all 40 overlong cases coincide with exhausted token budget and
  malformed `</function>` final output in this run.

## Raw log provenance

The raw logs are intentionally ignored by Git and retained separately:

- `logs/eval-bc-8b-baseline-64k.899659.out`
- `logs/eval-bc-8b-foldagent-64k.899660.out`
- `logs/eval-bc-8b-contextgraph-64k.899661.out`

