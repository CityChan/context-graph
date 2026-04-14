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
│   ├── context_graph.py        # Core ContextGraph: nodes, edges, graph ops, rewards
│   ├── graph_agent.py          # Agent loop with graph context (extends fold_agent)
│   ├── fold_agent.py           # Original FoldAgent (tree-based baseline)
│   ├── tool_spec.py            # Tool definitions (search + branch + graph ops)
│   ├── prompts.py              # Workflow prompts (search_graph, search_branch, etc.)
│   └── utils.py                # Agent/AgentContext classes, LLM client
├── envs/
│   ├── local_search.py         # Local search environment
│   └── search_server.py        # Qwen3-Embedding search service
├── scripts/
│   ├── train_graph.py          # ContextGraph training entry point
│   ├── train_fold.py           # FoldAgent training entry point (baseline)
│   ├── train_bc_contextgraph_4b.sh       # 1-node SLURM training
│   ├── train_bc_contextgraph_4b_2node.sh # 2-node SLURM training
│   └── test_train_graph_mini.sh          # Smoke test (0.6B, mock search)
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

## Training

**1. Start Search Server**

```bash
cd envs && python search_server.py \
  --model Qwen/Qwen3-Embedding-8B \
  --corpus Tevatron/browsecomp-plus-corpus \
  --corpus-embedding-dataset miaolu3/browsecomp-plus \
  --host 0.0.0.0 --port 8010
```

```bash
export LOCAL_SEARCH_URL="http://[IP-of-search-server]:8010"
export OPENAI_API_KEY="your-api-key"  # For LLM-based grading
```

**2. Download Training Data**

Download and decompress BrowseComp dataset: https://drive.google.com/file/d/1aX5xXAN5R-gLKd8A0AY-troxXJRawyAM/view?usp=sharing

**3. Train ContextGraph on BrowseComp**

```bash
# Single node (search service + training on same GPU)
sbatch scripts/train_bc_contextgraph_4b.sh

# 2-node (recommended: Node 0 = search + train, Node 1 = train)
sbatch scripts/train_bc_contextgraph_4b_2node.sh
```

**4. Train FoldAgent Baseline (for comparison)**

```bash
sbatch scripts/train_bc_qwen3_4b.sh
```

Both log to wandb project `context-graph` for direct comparison.

**5. Smoke Test**

```bash
sbatch scripts/test_train_graph_mini.sh  # 30min, Qwen3-0.6B, mock search
```

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
