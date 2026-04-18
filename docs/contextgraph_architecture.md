---
name: ContextGraph Architecture
description: ContextGraph project — graph-based working context management for LLM agents, built on top of FoldAgent. Architecture, file map, reward design, and key differences from FoldAgent.
type: project
---

## ContextGraph Overview

ContextGraph extends FoldAgent by modeling working context as a **dynamic graph** instead of a tree. Core idea: "Working context is not a sequence to be compressed, but a graph to be constructed, maintained, and transformed."

**Why:** FoldAgent uses tree-structured branches (main → independent branches). Branches can't share information, can't be merged, and search results sink into chat history without structure. ContextGraph adds cross-branch connections, merge/prune operations, and graph-aware rewards.

**How to apply:** When modifying FoldAgent code, ContextGraph files extend (not replace) the original. Both can run side-by-side with different workflow parameters.

## Project Location

- Local: `d:\Workspace\TACC-Work\context-graph\`
- TACC: `/work/09281/chc_1996/vista/context-graph`
- Conda env: `foldagent`

## File Map

### New Files (ContextGraph-specific)
| File | Purpose |
|------|---------|
| `agents/context_graph.py` | Core `ContextGraph` class: nodes, edges, graph ops, auto-heuristics, context reconstruction, reward computation |
| `agents/graph_agent.py` | Global-graph agent loop `process_item()` (registers `context_graph_agent`) |
| `agents/graph_agent_isolated.py` | Isolated subgraph variant (registers `context_graph_isolated_agent`) |
| `scripts/train_graph.py` | verl training entry point; registers both global + isolated agent loops |
| `scripts/train_alfworld_ctxgraph_30b_16node_48h.sh` | 16-node SLURM training on ALFWorld (Qwen3-30B-A3B-Thinking) |

### Modified Files
| File | Changes |
|------|---------|
| `agents/tool_spec.py` | Added `graph_tool()`: merge, add_edge, select, prune |
| `agents/prompts.py` | Added `search_graph` / `search_graph_multi` workflows + SEARCH_SYSTEM_PROMPT_GRAPH + SEARCH_USER_PROMPT_GRAPH |
| `verl/experimental/agent_loop/__init__.py` | Registered `ContextGraphAgentLoop` |

## ContextGraph vs FoldAgent — Key Differences

### Data Structure
- **FoldAgent**: `agent = dict()` with Agent objects, no structural tracking
- **ContextGraph**: `ContextGraph(tokenizer)` tracks all nodes/edges; `agent = dict()` kept for verl compatibility

### Action Space
| Tool | FoldAgent | ContextGraph |
|------|-----------|-------------|
| search, open_page, finish | Yes | Yes |
| branch, return | Yes | Yes + creates subtask node + decomposition edge |
| **merge** | No | Combine multiple nodes → summary node |
| **add_edge** | No | Connect any two nodes (causal/semantic/temporal) |
| **select** | No | Change active focus node |
| **prune** | No | Remove low-value node from active context |

### Graph Operations: Who Decides?

Two layers of decision-making:

1. **Automatic (heuristic)** — runs every turn in agent loop:
   - `auto_prune_low_value(max_active=20)`: prune lowest-value observation nodes when graph exceeds 20 active
   - `auto_connect_semantic(threshold=3)`: add semantic edges between nodes from different subtasks sharing ≥3 keywords
   - `auto_merge_similar(threshold=3)`: merge when a subtask has ≥3 observation children

2. **LLM-driven** — model generates `<function=merge>`, `<function=prune>`, etc. tool calls. Learned via RL process rewards.

Auto-heuristics solve cold start; LLM-driven ops are the RL training target.

### Observation Format
- **FoldAgent**: Pure text ("Branch has finished its task...")
- **ContextGraph**: Text + graph state appended to every observation:
  ```
  [Graph] 5/7 nodes active, 6 edges, ops=8, focus=[n1]
    [n1]* query: What is the capital of France? [12tok]
    [n2] subtask: Verify capital city [45tok]
    ...
    Edges: n1->n2(decomp), n2->n3(causal)
  ```

### Context Tracking
- **FoldAgent**: search/open_page results go into chat history only
- **ContextGraph**: automatically added as observation nodes with typed edges; can be merged/pruned/referenced later

### Reward Design

Total: `r = r_task + r_graph - r_cost`

| Component | Formula | Condition |
|-----------|---------|-----------|
| r_task | Binary 0/1 from env judge | Always |
| compactness | (n_folded + n_pruned) / n_total | Only if task succeeds |
| structural | n_cross_edges / n_active | Only if task succeeds |
| merge_bonus | min(n_summaries * 0.1, 0.3) | Only if task succeeds |
| prune_bonus | min(n_pruned * 0.05, 0.15) | Only if task succeeds |
| cost_penalty | λ_cost * operation_count | Always (prevents reward hacking) |

Per-turn process rewards (task success only):
- merge turn: +0.2
- prune turn: +0.1
- add_edge turn: +0.1
- select turn: 0 (neutral)
- verbose/no-structure/CJK/tool-error: same as FoldAgent

**Key design principle:** Graph rewards conditioned on task success — prevents agent from hacking reward by doing graph ops without solving the task.

### Backward Compatibility
When agent uses NO graph ops (only branch/search/finish), ContextGraph degrades to FoldAgent:
- Graph auto-adds nodes but no merging/pruning if below threshold
- r_graph ≈ 0, r_cost ≈ small
- Output format identical (same AgentLoopOutput)

## Storage
ContextGraph is **purely in-memory** (Python dict + list). No database, no file I/O.
- Lifetime = one episode (one question, start to finish)
- Graph discarded after reward computation
- Only reward metrics persisted via `extra_fields['graph_rewards']` → wandb

## Config Parameters
| Param | Default | Description |
|-------|---------|-------------|
| `workflow` | `search_graph` | Selects graph prompt + tools |
| `process_reward` | `[flat,scope,graph]` | Enable graph process rewards |
| `lambda_compact` | 0.1 | Weight for graph shaping rewards |
| `lambda_cost` | 0.005 | Weight for operation cost penalty |

## How to Run
```bash
# ContextGraph isolated (16-node, Qwen3-30B-A3B-Thinking, ALFWorld)
sbatch scripts/train_alfworld_ctxgraph_30b_16node_48h.sh

# FoldAgent baseline (16-node, same model + ALFWorld)
sbatch scripts/train_alfworld_fold_30b_16node_48h.sh
```

wandb project: `context-graph`
