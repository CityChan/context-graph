# ContextGraph: RL-based Graph Construction for Working Context Management

> Extending [FoldAgent](https://arxiv.org/pdf/2510.11967) (Context-Folding) from tree-structured branching to **graph-structured working context**.

**Core idea:** Working context is not a sequence to be compressed, but a graph to be constructed, maintained, and transformed.

See [docs/contextgraph_architecture.md](docs/contextgraph_architecture.md) for full architecture details, reward design, and FoldAgent vs ContextGraph comparison.

---

## What's New over FoldAgent

| Feature | FoldAgent | ContextGraph |
|---------|-----------|-------------|
| Context structure | Tree (main → branches) | Full graph (any-to-any edges) |
| Branch results | Independent, text only | Stored as graph nodes with typed edges |
| Cross-branch links | Not possible | `add_edge` connects any two nodes |
| Consolidation | Manual re-reading | `merge` combines nodes into summaries |
| Context cleanup | Session overflow truncation | `prune` removes dead-end nodes |
| Focus control | Always on main agent | `select` shifts attention to any node |
| Search tracking | Not tracked | Auto-added as observation nodes |
| Rewards | task + scope + cjk | + compactness + structural + merge/prune bonus - cost |
| Graph decisions | N/A | Two-layer: auto-heuristics + LLM tool calls (learned via RL) |

---

## Key Files

```
context-graph/
├── agents/
│   ├── context_graph.py            # Core ContextGraph: nodes, edges, graph ops, rewards
│   ├── graph_agent.py              # Global ContextGraph agent loop
│   ├── graph_agent_isolated.py     # Isolated (per-branch subgraph) agent loop
│   ├── fold_agent.py               # Original FoldAgent (tree-based baseline)
│   ├── react_agent.py              # ReAct baseline
│   ├── tool_spec.py                # Tool definitions (search + branch + graph ops)
│   ├── prompts.py                  # Workflow prompts (search_graph, search_branch, etc.)
│   ├── verifier.py                 # Reward verifier
│   └── utils.py                    # Agent/AgentContext classes, LLM client
├── envs/
│   ├── local_search.py             # Local search client
│   ├── search_server.py            # Qwen3-Embedding search service
│   ├── alfworld_env.py             # ALFWorld environment
│   └── repo_env.py / repo_server.py # SWE repo environment
├── scripts/
│   ├── train_graph.py              # ContextGraph training entry point
│   ├── train_fold.py               # FoldAgent training entry point (baseline)
│   ├── eval_bc.py                  # BrowseComp evaluation
│   ├── train_bc_8b_8node.sh                # FoldAgent 8-node training (BrowseComp)
│   ├── train_bc_8b_8node_contextgraph.sh   # ContextGraph 8-node training (BrowseComp)
│   ├── make_alfworld_data.py / make_multihop_data.py # Dataset builders
│   └── multihop_search_server.py   # Multihop QA search service
├── verl/                       # Vendored verl framework (with ARM/vllm compat fixes)
│   ├── experimental/agent_loop/  # Agent loop base + registry
│   └── trainer/ppo/              # FoldGRPO algorithm
└── docs/
    └── contextgraph_architecture.md  # Full architecture documentation
```

---

## Graph-MDP Formulation

**State:** s_t = (observation, graph G_t, active_node)

**Actions:** search, open_page, branch, finish, **merge**, **add_edge**, **select**, **prune**

**Reward:** r = r_task + λ₁ · (compactness + structural + merge_bonus + prune_bonus) − λ₂ · cost
- Graph shaping rewards only apply when task succeeds (prevents reward hacking)
- Cost penalty always applies

---

## Setup

**1. Create conda env and install Python deps**

```bash
conda create -n contextgraph python=3.10 -y
conda activate contextgraph
pip install torch  # match your CUDA version
pip install -r requirements.txt
bash scripts/setup_env.sh   # installs extras not pinned in requirements.txt
```

**2. Install vLLM** (rollout backend; needs a GPU node with CUDA toolkit)

```bash
pip install vllm
```

On TACC Vista (ARM aarch64 / GH200), build on a compute node:

```bash
idev -p gh -N 1 -n 1 -t 01:00:00
conda activate contextgraph
pip install vllm
```

**3. (Optional) Flash Attention** — required for 8B+ models with long sequences

```bash
TORCH_CUDA_ARCH_LIST="9.0a" pip install flash-attn --no-build-isolation
```

**4. Environment variables**

```bash
export LOCAL_SEARCH_URL="http://[search-server-host]:8010"  # set after starting search server (see Training step 1)
export OPENAI_API_KEY="..."                                  # for LLM-based grading
export WANDB_API_KEY="..."                                   # optional, for run logging
```

---

## Training

**1. Start Search Server**

```bash
cd envs && python search_server.py \
  --model Qwen/Qwen3-Embedding-8B \
  --corpus Tevatron/browsecomp-plus-corpus \
  --corpus-embedding-dataset miaolu3/browsecomp-plus \
  --host 0.0.0.0 --port 8010
```

Then point `LOCAL_SEARCH_URL` at it (see Setup step 4).

**2. Download Training Data**

Download and decompress BrowseComp dataset: https://drive.google.com/file/d/1aX5xXAN5R-gLKd8A0AY-troxXJRawyAM/view?usp=sharing

**3. Train ContextGraph on BrowseComp**

```bash
sbatch scripts/train_bc_8b_8node_contextgraph.sh
```

**4. Train FoldAgent Baseline (for comparison)**

```bash
sbatch scripts/train_bc_8b_8node.sh
```

Both log to wandb project `context-graph` for direct comparison.

---

## Evaluation

**ContextGraph Agent:** `workflow=search_graph`
```bash
python scripts/eval_bc.py \
  --workflow search_graph \
  --model_name Qwen/Qwen3-4B-Instruct-2507 \
  --num_workers 32 \
  --prompt_length 8192 \
  --response_length 32768 \
  --max_turn 200 \
  --max_session 10 \
  --output_dir results
```

**FoldAgent Baseline:** `workflow=search_branch`
```bash
python scripts/eval_bc.py --workflow search_branch [...]
```

**ReAct Agent:** `workflow=search`
```bash
python scripts/eval_bc.py --workflow search [...]
```

---

## Cite

```
@article{sun2025scaling,
  title   = {Scaling Long-Horizon LLM Agent via Context-Folding},
  author  = {Sun, Weiwei and Lu, Miao and Ling, Zhan and Liu, Kang and Yao, Xuesong and Yang, Yiming and Chen, Jiecao},
  journal = {arXiv preprint arXiv:2510.11967},
  year    = {2025},
}
```

---

## Acknowledgements

This implementation is based on [FoldAgent](https://github.com/sunnweiwei/FoldAgent) and [verl](https://github.com/volcengine/verl).
