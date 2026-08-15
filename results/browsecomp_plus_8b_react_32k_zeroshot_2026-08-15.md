# BrowseComp-Plus Qwen3-8B ReAct 32K zero-shot evaluation

Frozen on 2026-08-15 from the full 150-sample baseline evaluations. The
headline zero-shot result is **25/150 = 0.1667 (16.67%)** from Slurm job
`913769`.

## Evaluation configuration

- Method: vanilla ReAct (`react_agent` with `search_base`)
- Model: `Qwen/Qwen3-8B`
- Test split: `data/bc_test.parquet`, all 150 samples
- Topology: four Vista GH nodes (one dedicated search node and three trainer nodes)
- Prompt length: 8,192 tokens
- Response length: 32,768 tokens
- Active context length: 40,960 tokens
- Maximum sessions: 10
- Maximum turns: 100
- Session timeout: 600 seconds
- Validation: greedy, one trajectory per sample
- Repository head used by the headline run: `50fccdb`

## Result

| Role | Slurm job | Correct | Task accuracy | Session timeout | Status |
|---|---:|---:|---:|---:|---|
| Headline run | 913769 | 25/150 | 0.1667 | 600 s | All samples judged; auxiliary aggregation failed after generation |
| Earlier diagnostic | 913501 | 27/150 | 0.1800 | 3,600 s | All samples judged; auxiliary aggregation failed after generation |

The headline score is the latest script-default result. The earlier diagnostic
is retained for provenance but is not a strict replicate because its session
timeout differs. Do not report the arithmetic mean as a repeated-run estimate.

Both jobs completed generation and reward judging for all 150 samples. They
then exited with a missing-auxiliary-metric aggregation error, so task accuracy
was recovered directly by counting the 150 `[Judged] score=` records. The two
aggregation paths were fixed by commits `f47dadc` and `98367e5`.

## Raw log provenance

The raw logs are ignored by Git and retained separately:

- `logs/eval-bc-8b-base-zeroshot.913769.out`
- `logs/eval-bc-8b-base-zeroshot.913769.err`
- `logs/eval-bc-8b-react-32k.913501.out`
- `logs/eval-bc-8b-react-32k.913501.err`
