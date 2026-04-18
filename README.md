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

**3. (Optional) Flash Attention** — required for 8B+ models with long sequences

```bash
TORCH_CUDA_ARCH_LIST="9.0a" pip install flash-attn --no-build-isolation
```

**4. Environment variables**

```bash
export LOCAL_SEARCH_URL="http://[search-server-host]:8010"  # set after starting a search server (BrowseComp / Multi-hop)
export OPENAI_API_KEY="..."                                  # for LLM-based grading
export WANDB_API_KEY="..."                                   # optional, for run logging
```

---

## Experiments to Run

We mirror the FoldAgent paper's experimental structure (Sun et al. 2025, [arXiv:2510.11967](https://arxiv.org/abs/2510.11967)), adapted for the ContextGraph extension across our three environments.

### Plan

| Phase | Goal | Status |
|-------|------|--------|
| **0** | Smoke test 30B model on ALFWorld (env validation) | scripted |
| **1** | FoldAgent baseline: Qwen3-30B-A3B-Thinking-2507 on ALFWorld | scripted |
| **2** | ContextGraph (isolated) on ALFWorld at same settings | scripted |
| **3** | Repeat 1+2 on BrowseComp and Multi-hop QA | TBD |
| **4** | Ablations: auto-merge on/off, isolated vs global, prompt-length matched, FoldGRPO vs vanilla GRPO | TBD |
| **5** | Behavior analysis: Finish rate, Main Len, Scope, # Branch, # graph ops, # cross-edges (mirror paper Table 2 + graph extras) | TBD |

### Hyperparameters (Phase 1+, adapted from paper §5)

| Setting | Value | Notes |
|---------|-------|-------|
| Base model | `Qwen/Qwen3-30B-A3B-Thinking-2507` | MoE, 30B total / 3B active |
| Optimizer | Adam, lr 5e-6, weight_decay 0.1 | paper §5 |
| KL penalty | 0.001 against frozen reference | paper §5 |
| Context window | 32K (`response_length`) | paper §5 |
| Branch threshold | 8K (`branch_len` ≈ paper's "context penalty threshold") | paper §5 |
| GRPO group size | n=8 | paper §5 |
| Batch size | 32 prompts × 8 samples = 256 trajectories / step | paper §5 |
| Hardware | 16 × GH200 (FSDP) | adapted for TACC Vista |
| Training budget | 48 h (target ~500 steps) | |

### Caveats vs paper

The paper uses **Slime + INT4 + QAT + TIS (clip 2.0) + token-in-token-out**.
Our verl-based implementation runs in **BF16 with standard tokenization between turns**.
INT4/QAT and TIS are not (yet) implemented in this repo — these gaps are documented
so any quantitative deviation from the paper is expected.

---

## Training

All runs log to wandb project `context-graph` for direct comparison.

### BrowseComp

**1. Start the search server** (Qwen3-Embedding-8B over the BrowseComp+ corpus)

```bash
cd envs && python search_server.py \
  --model Qwen/Qwen3-Embedding-8B \
  --corpus Tevatron/browsecomp-plus-corpus \
  --corpus-embedding-dataset miaolu3/browsecomp-plus \
  --host 0.0.0.0 --port 8010
```

Then point `LOCAL_SEARCH_URL` at it (see Setup step 4).

**2. Download the dataset**

Download and decompress: https://drive.google.com/file/d/1aX5xXAN5R-gLKd8A0AY-troxXJRawyAM/view?usp=sharing

**3. Launch training**

```bash
sbatch scripts/train_bc_8b_8node_contextgraph.sh   # ContextGraph
sbatch scripts/train_bc_8b_8node.sh                # FoldAgent baseline
```

---

### ALFWorld

No search server needed — uses local TextWorld game files.

**1. Install ALFWorld and download games**

```bash
pip install textworld alfworld
alfworld-download                                  # writes to ~/.cache/alfworld
```

**2. Generate parquet from real game files**

```bash
python scripts/make_alfworld_data.py --n_train 300 --n_val 80
# add --hard for MemexRL-style (admissible commands hidden from agent)
```

**3. Smoke test the environment** (2 nodes, 1 hour, validates 30B model + Ray + 3 RL steps)

```bash
sbatch scripts/test_alfworld_30b_2node_1h.sh
```

**4. Launch production training** (16 nodes × 1 GH200, 48 hours)

```bash
sbatch scripts/train_alfworld_fold_30b_16node_48h.sh        # FoldAgent baseline
sbatch scripts/train_alfworld_ctxgraph_30b_16node_48h.sh    # ContextGraph (isolated)
```

For interactive iteration on a small (4B) model:

```bash
bash scripts/idev_fold.sh        # FoldAgent on ALFWorld (Qwen3-4B, 20 steps)
bash scripts/idev_ctxgraph.sh    # ContextGraph isolated on ALFWorld (Qwen3-4B, 20 steps)
```

---

### Multi-hop QA

Synthetic 2–3 hop benchmark with a built-in knowledge base — no external corpus.

**1. Start the multi-hop search server** (TF-IDF over a small KB, port 18999)

```bash
python scripts/multihop_search_server.py
export LOCAL_SEARCH_URL="http://localhost:18999"
```

**2. Generate parquet**

```bash
python scripts/make_multihop_data.py --n_train 300 --n_val 80
```

**3. Launch training**

Use `train_fold.py` / `train_graph.py` directly with `data.train_files=data/multihop_train.parquet` and `data.val_files=data/multihop_test.parquet` (no canned sbatch wrapper — model the args after the BrowseComp scripts).

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
