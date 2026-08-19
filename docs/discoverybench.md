# DiscoveryBench evaluation

This integration evaluates the official `allenai/discoverybench` real test
split with the benchmark's Hypothesis Match Score (HMS). It is evaluation-only:
gold hypotheses remain in `extra_info` and are never included in the agent
prompt. ReAct, FoldAgent, and ContextGraph share the same data and scorer.

## Prepare data

From the project root, download the official Hugging Face snapshot and build
the three workflow-specific parquet files:

```bash
HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 python scripts/make_discoverybench_data.py --out-dir data --dataset-type real --split test
```

The real test split contains 239 queries. The source tables and metadata stay
under `data/discoverybench`; parquet rows point to those files rather than
duplicating them.

## Four-node smoke

Inside a four-node GH `idev` allocation, load judge credentials (the existing
`$WORK/.openai_env` is discovered automatically), then run one query:

```bash
DISCOVERYBENCH_METHOD=react DISCOVERYBENCH_VAL_MAX_SAMPLES=1 bash scripts/smoke_discoverybench_qwen3_30b_instruct_4node.sh
```

Set the method to `fold` or `ctxgraph` for the other agents. By default smoke
still uses real HMS; set `DISCOVERYBENCH_REAL_EVAL=0` only when diagnosing the
file/JSON pipeline. A format-only score is not a benchmark result.

## Full test evaluation

Submit one formal job per method:

```bash
DISCOVERYBENCH_METHOD=react sbatch scripts/eval_discoverybench_qwen3_30b_instruct_8node.sh
DISCOVERYBENCH_METHOD=fold sbatch scripts/eval_discoverybench_qwen3_30b_instruct_8node.sh
DISCOVERYBENCH_METHOD=ctxgraph sbatch scripts/eval_discoverybench_qwen3_30b_instruct_8node.sh
```

For the matched Qwen3-8B suite, each method uses four GH nodes and requests an
eight-hour walltime. Submit all three methods with:

```bash
bash scripts/submit_eval_discoverybench_qwen3_8b_4node.sh
```

Select a subset with `DISCOVERYBENCH_METHODS`, for example:

```bash
DISCOVERYBENCH_METHODS=react bash scripts/submit_eval_discoverybench_qwen3_8b_4node.sh
```

The sample cap and requested walltime can be overridden with
`DISCOVERYBENCH_VAL_MAX_SAMPLES` and `DISCOVERYBENCH_TIME_LIMIT`.

Per-query predictions, gold references, decompositions, matches, and HMS
components are written to `$SCRATCH/discoverybench_results/<job_id>/`. Validation
generations are also retained. Report mean `val/hms_score` (equivalently
`val/reward`) over all 239 queries, plus JSON-valid rate and judge-error rate.
The OpenAI default is the upstream evaluator's `gpt-4-1106-preview`; Azure uses
the deployment named by `AZURE_OPENAI_DEPLOYMENT_NAME`, which is recorded in
every audit result. Use one fixed judge deployment for all method comparisons.

Summarize a completed job without parsing Ray logs:

```bash
python scripts/summarize_discoverybench_results.py "$SCRATCH/discoverybench_results/<job_id>"
```
