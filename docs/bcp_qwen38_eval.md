# Qwen3.8-27B BC-P evaluation on two Vista idev allocations

Use one existing four-node allocation per method. This does not submit jobs or
train a model. Each allocation runs one BC-P retriever and three independent
BF16 model replicas (one per GH200, TP=1), with deterministic disjoint row shards.
Default concurrency is two tasks per replica. Start with eight uniformly selected
questions (seed 42). Use `SAMPLES=-1` for all rows after the smoke succeeds.

In the first allocation, from the updated checkout:

```bash
bash scripts/eval_bcp_qwen38_4node_idev.sh foldagent
```

In the second allocation:

```bash
bash scripts/eval_bcp_qwen38_4node_idev.sh contextgraph
```

Both allocations must use the same commit, model snapshot, data file, sample
count and seed. Do not update the shared checkout while either evaluation runs.
`EXPECTED_JOB_ID` optionally checks the intended allocation. Each invocation
creates a fresh output directory; existing directories are rejected.

The downloaded `Qwen/Qwen3.8-27B` checkpoint is resolved offline from
`$SCRATCH/hf_cache/hub`, or supplied through `MODEL_PATH` (must remain under
scratch). The retriever retains its separate `/work/.../vista/cache` cache.
No packages are installed. Model serving uses `SERVER_CONDA_ENV=deepseek_v4`;
the evaluator and retriever use `AGENT_CONDA_ENV=cxtgraph`. Both environments
must already have compatible dependencies. The script checks tokenizer evidence
retention, model loading, generated token IDs and structured JSON responses.
Actual architecture/kernel compatibility and memory headroom require the Vista run.

The two methods use the same 32K budget (8K prompt + 24K response), 100 turns,
10 sessions, 2048 tokens per turn, greedy decoding, thinking enabled with retained
thinking history, and 1024-token finalizer reserve. ContextGraph uses its isolated
structured controller, `MEMORY_MODE=repaired`, consolidation every five turns,
and active-node cap 12. `MEMORY_MODE=legacy` is an explicit alternative, not the
default here. StructMem and answer/repeat diagnostic interventions are disabled.
These are new 27B evaluation results, not a reproduction of older 8B controls.

The token completion endpoint receives the exact context assembled by the agent,
and returns exact generated IDs, including reasoning. No chat API reasoning field
is silently dropped. Controller turns use guided JSON. The model endpoint receives
no OpenAI judge credentials. Judge credentials are loaded from the usual `.openai_env`;
`JUDGE_BASE_URL` optionally selects a judge endpoint. Default judge is GPT-5-nano.

Outputs are under `outputs/bcp-qwen38-METHOD-JOB-TIMESTAMP/`:

- `suite.log`, `search.log`, `model-0.log` through `model-2.log`, `eval-*.log`.
- `manifest-*.json`: source hash, row indices, Git commit, snapshot and effective configuration.
- `results-*.jsonl`: append-only task outcomes, preserved even if the allocation ends.
- `trajectory-*.json` and `requests-*.jsonl`: trajectories and exact token requests.
- `summary.json`: task success, finish count, execution and judge parsing failures.

`BCP_EVAL_COMPLETE` is printed only after all shards finish, expected rows are
accounted for, and no execution/judge parsing failures remain. A missing marker
does not prove a crash; inspect logs and allocation time. Report task accuracy,
not shaped agent reward or execution-completion count.

Full evaluation command (in each allocation, choose the method):

```bash
SAMPLES=-1 bash scripts/eval_bcp_qwen38_4node_idev.sh foldagent
```
