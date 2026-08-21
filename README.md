# ContextGraph

ContextGraph extends FoldAgent by managing an agent's working context as a graph instead of a tree. Search results, branches, summaries, and selected focus nodes become explicit graph state that can be merged, linked, selected, and pruned during long-horizon agent rollouts.

The repo is currently scoped to four benchmark tracks:

| Track | Role | Status |
| --- | --- | --- |
| BrowseComp-Plus | Main search/research QA benchmark | Primary track |
| GAIA | General assistant/search benchmark | Active text-only integration |
| ScienceAgentBench (SAB) | Code-execution benchmark | Active evaluation track |
| ALFWorld | Stateful embodied-text benchmark | Experimental/diagnostic |

Multi-hop QA wrappers for HotpotQA, MuSiQue, and 2WikiMultiHopQA were removed from the active codebase. Shared search infrastructure remains because BrowseComp-Plus still uses it.

## Core Files

| Path | Purpose |
| --- | --- |
| `agents/context_graph.py` | In-memory graph state, graph ops, and graph reward accounting |
| `agents/graph_agent.py` | Global ContextGraph agent loop |
| `agents/graph_agent_isolated.py` | Isolated per-branch ContextGraph agent loop |
| `agents/fold_agent.py` | FoldAgent baseline |
| `agents/react_agent.py` | ReAct baseline for search-style tasks |
| `agents/react_agent_code.py` | ReAct baseline for SAB code-execution tasks |
| `agents/prompts.py` | Search and ALFWorld prompts/tool instructions |
| `agents/prompts_code.py` | SAB code-agent prompts |
| `envs/search_server.py` | BrowseComp-Plus embedding search service |
| `envs/local_search.py` | Local search client used by search agents |
| `envs/scienceagent_env.py` | SAB environment wrapper |
| `envs/scienceagent_sandbox.py` | Stateful restricted Python sandbox for SAB |
| `envs/alfworld_env.py` | ALFWorld TextWorld wrapper |
| `scripts/train_graph.py` | ContextGraph training entry point |
| `scripts/train_fold.py` | FoldAgent training entry point |
| `scripts/train_baseline.py` | ReAct/baseline training entry point |
| `scripts/eval_bc.py` | BrowseComp-Plus evaluation entry point |
| `scripts/make_gaia_data.py` | GAIA parquet builder |
| `scripts/eval_gaia.py` | GAIA API-based evaluation entry point |
| `scripts/train_sab.py` | SAB evaluation/training entry point |

## Setup

```bash
conda create -n cxtgraph python=3.10 -y
conda activate cxtgraph
pip install torch
pip install -r requirements.txt
bash scripts/setup_env.sh
```

On TACC Vista/GH200, use the cluster CUDA/vLLM environment already configured in the sbatch scripts. The scripts assume the `cxtgraph` conda env and project path `/work/09281/chc_1996/vista/context-graph` unless overridden.

## BrowseComp-Plus

Start the embedding search server:

```bash
cd envs && python search_server.py --model Qwen/Qwen3-Embedding-8B --corpus Tevatron/browsecomp-plus-corpus --corpus-embedding-dataset miaolu3/browsecomp-plus --host 0.0.0.0 --port 8010
```

Point agents at the server:

```bash
export LOCAL_SEARCH_URL="http://<search-server-host>:8010"
```

Representative zero-shot/eval scripts:

```bash
bash scripts/eval_bc_baseline_8b_4node_zeroshot.sh
bash scripts/eval_bc_ctxgraph_30b_8node_zeroshot.sh
bash scripts/eval_bc_foldagent_30b_8node_zeroshot.sh
```

## GAIA

GAIA is gated on HuggingFace. Login before building data:

```bash
huggingface-cli login
```

Build text-only validation parquets:

```bash
python scripts/make_gaia_data.py --split validation --out-dir data
```

This writes:

```text
data/gaia_validation.parquet
data/gaia_validation_branch.parquet
data/gaia_validation_graph.parquet
```

The first integration skips rows with file attachments by default because the current GAIA agent path exposes search/open-page tools, not image/OCR/spreadsheet/file tools. Use `--include-files` only for debugging metadata flow.

Run a small API-based smoke eval:

```bash
python scripts/eval_gaia.py --data-path data/gaia_validation_graph.parquet --workflow search_graph --max-samples 8 --num-workers 2 --local-search-url http://localhost:8010
```

## ScienceAgentBench

Build SAB parquets after downloading the upstream CSV and benchmark package:

```bash
python scripts/make_sab_data.py --csv data/ScienceAgentBench.csv --benchmark-dir data/sab_benchmark --out-dir data
```

Run the 8B ReAct smoke eval:

```bash
SAB_VAL_MAX_SAMPLES=8 SAB_DEBUG_IO=1 SAB_DUMP_VALIDATION=1 SAB_NO_OUTPUT_HINT_AFTER=2 SAB_RESPONSE_LENGTH=12288 SAB_TURN_MAX_NEW_TOKENS=512 bash scripts/eval_sab_react_8b_4node_smoke.sh
```

The same hardened 4-node harness can run all three SAB methods by setting
`SAB_METHOD` to `react`, `fold`, or `ctxgraph`. It selects the matching agent
loop, workflow parquet, and process-reward configuration automatically:

```bash
SAB_METHOD=fold SAB_REAL_EVAL=1 SAB_VAL_MAX_SAMPLES=1 bash scripts/eval_sab_react_8b_4node_smoke.sh
SAB_METHOD=ctxgraph SAB_REAL_EVAL=1 SAB_VAL_MAX_SAMPLES=1 bash scripts/eval_sab_react_8b_4node_smoke.sh
```

Other SAB entry points live under `scripts/eval_sab_*.sh`.

## ALFWorld

Install game dependencies and download game files:

```bash
pip install textworld alfworld
alfworld-download
```

Build real/hard parquets:

```bash
python scripts/make_alfworld_data.py --mode real --n_train 300 --n_val 80
python scripts/make_alfworld_data.py --mode hard --n_train 300 --n_val 80
```

Current 8B diagnostic script:

```bash
bash scripts/train_alfworld_ctxgraph_8b_4node_30step.sh
```

For an action-only ReAct diagnostic on the same script:

```bash
ALFWORLD_TRAIN_MODULE=scripts.train_baseline ALFWORLD_AGENT_LOOP=react_agent ALFWORLD_WORKFLOW=alfworld ALFWORLD_PROCESS_REWARD='[flat]' ALFWORLD_VAL_ONLY=True ALFWORLD_VAL_MAX_SAMPLES=8 ALFWORLD_MAX_TURN=40 ALFWORLD_VAL_MAX_TURN=40 ALFWORLD_TURN_MAX_NEW_TOKENS=128 bash scripts/train_alfworld_ctxgraph_8b_4node_30step.sh
```

## Tests

Local smoke checks:

```bash
python -m tests.smoke_sab_sandbox
python -m tests.smoke_gaia_data
python -m py_compile envs/alfworld_env.py
bash -n scripts/eval_sab_react_8b_4node_smoke.sh
bash -n scripts/train_alfworld_ctxgraph_8b_4node_30step.sh
```

## ContextGraph SFT data

The open-source teacher pipeline uses `deepseek-ai/DeepSeek-V4-Flash-0731`.
It expects the model under `$SCRATCH` (either a direct directory or the normal
Hugging Face cache layout), eight GH200 nodes for a TP=8 vLLM server, and one
additional node for BrowseComp-Plus retrieval. The DeepSeek server runs in a
dedicated `deepseek_v4` conda environment with vLLM 0.25 or newer; the agent
runner remains in `cxtgraph`.

Submit a 25-sample pilot:

```bash
PREFLIGHT_ONLY=1 bash scripts/generate_ctxgraph_sft_deepseek_v4_flash_0731_9node.sh
sbatch scripts/generate_ctxgraph_sft_deepseek_v4_flash_0731_9node.sh
```

Override an explicit checkpoint directory or increase the deterministic shard:

```bash
MODEL_PATH=$SCRATCH/models/DeepSeek-V4-Flash-0731 START_INDEX=0 MAX_SAMPLES=100 sbatch scripts/generate_ctxgraph_sft_deepseek_v4_flash_0731_9node.sh
```

Only correct, finished, non-overlong trajectories with no invalid graph calls
and at least one successful structural graph operation are retained.

For a quick smoke inside an existing 4-node Vista `idev` allocation, use the
single-GPU `Qwen/Qwen3.6-27B` teacher. The smoke uses one node for vLLM and one
for retrieval; the other two allocated nodes remain idle. It evaluates two
BrowseComp train questions by default:

```bash
MAX_SAMPLES=2 bash scripts/smoke_generate_ctxgraph_sft_qwen3_6_27b_4node_idev.sh
```

The output is written below
`$SCRATCH/contextgraph_sft/qwen3_6_27b_smoke/$SLURM_JOB_ID`. The server requires
vLLM 0.19 or newer; by default it reuses the `deepseek_v4` server environment.

## Documentation

Architecture and reward design are documented in `docs/contextgraph_architecture.md`. SAB-specific design notes are in `docs/design_scienceagentbench_ctxgraph.md`.
