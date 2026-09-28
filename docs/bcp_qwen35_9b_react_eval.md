# Qwen3.5-9B BC-P ReAct batch evaluation

Submit from a Vista login node; no idev allocation is needed:

```bash
cd /work/09281/chc_1996/vista/context-graph && git pull --ff-only origin master && mkdir -p logs && sbatch scripts/eval_bcp_qwen35_9b_react_4node.sbatch
```

This requests four `gh` nodes for **one hour**, using account `AST24021`.
Override the account at submission with `sbatch -A YOUR_ACCOUNT ...` if needed.
The job runs ReAct on all rows of `data/bc_test.parquet` (150 in the current
dataset), explicitly fixes `BENCHMARK=bcp`, and uses `Qwen/Qwen3.5-9B` from
the existing scratch HF cache. No download or training is performed.

The launcher preserves the existing 32K context, greedy decoding, seed 42,
two workers per model replica, local BC-P retrieval, and judge settings.
It uses one retriever node and three model replicas, with separate result shards.
The time limit includes service startup and evaluation; a Slurm allocation does
not guarantee completion within that limit.

Slurm logs: `logs/bcp-qwen35-9b-react.JOBID.out` and `.err`.
Full artifacts: `outputs/bcp-qwen35-9b-react-JOBID-TIMESTAMP/`.
Verify `Benchmark=bcp`, `bc_test.parquet`, 150 selected rows, and finally
`BCP_EVAL_COMPLETE method=react` with a complete `summary.json`.

Do not update the shared code checkout while the evaluation is running.
