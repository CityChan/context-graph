# Design Doc: ContextGraph on ScienceAgentBench

**Goal**: Adapt our search-based ContextGraph (BC-Plus) to code-execution
domain (ScienceAgentBench, 102 eval-only tasks). Zero-shot only. No training.

**Paper claim to test**: ContextGraph's architectural state (auto-bind edges,
graph projection, sub-task branches) provides task-success lift in domains
beyond search. Negative control: FoldAgent under same setup should NOT
generalize as well, because it lacks structural state that captures variable
lineage in code trajectories.

**Three agent variants to compare** (all zero-shot, all use same backbone):
1. `react_agent_code` — vanilla ReAct, single thread, no branches, no graph
2. `fold_agent_code` — branch + return, no graph
3. `code_graph_isolated_agent` — branch + return + graph state + auto-bind + uniqueness

---

## 1. Tool Spec (minimal)

Only one external tool: `python_exec`. Replaces BC-Plus's `search` + `open_page`.

```python
# agents/tool_spec_code.py (NEW FILE)

CODE_EXEC_TOOL = {
    "name": "python_exec",
    "description": (
        "Execute Python code in a persistent sandbox. State (imported modules, "
        "defined variables, loaded data) PERSISTS across calls within the same "
        "trajectory. Returns stdout and stderr (truncated to 2KB). Working "
        "directory contains input data files; output files should be written to "
        "`pred_results/`. Errors do not abort the trajectory — agent can debug."
    ),
    "parameters": {
        "code": {
            "type": "string",
            "description": "Python code to execute. Multi-line OK. Imports persist.",
        }
    },
}
```

Branch / merge / add_edge / select / prune / return / finish — unchanged from
BC-Plus, registered the same way. The agent loop just SWAPS the action space:
search/open_page replaced by python_exec.

---

## 2. Agent Loop A: `react_agent_code` (baseline)

Single-threaded ReAct loop:

```
turn 1:  agent emits python_exec(code="import scanpy as sc; ...")
         env returns stdout="loaded h5ad; (20000, 33538) cells x genes"
turn 2:  agent emits python_exec(code="sc.pp.filter_genes(adata, ...)")
         env returns stdout="filtered to (20000, 18432)"
...
turn 7:  agent emits finish(message="saved pred_results/...png")
```

No state structure beyond linear history. **Identical implementation to
`react_agent` in BC-Plus**, only difference is the registered tools.

File change: copy `agents/react_agent.py` → `agents/react_agent_code.py`,
register `python_exec` instead of `search` in the tool dispatch.

---

## 3. Agent Loop B: `fold_agent_code` (branch + return)

Same as BC-Plus's fold_agent, with python_exec instead of search. Branches
delegate sub-tasks; on return, summary is injected into parent.

Example trajectory on task #69 (UMAP single-cell):

```
[main t1] python_exec("import scanpy; adata = sc.read('hca/.h5ad')")
                       → returns shape info
[main t2] branch("Explore data", "Inspect adata schema: obs columns, var columns, X dtype")
        ├── [b1.t1] python_exec("print(adata.obs.columns)") → cell types found
        ├── [b1.t2] python_exec("print(adata.var.head())")  → gene metadata
        └── [b1.return msg="cell type column is 'cell_type'; 18432 genes after filter"]
[main t3] python_exec("sc.pp.filter_genes(adata, min_cells=10)") → ...
[main t4] python_exec("sc.tl.pca(adata, n_comps=30)") → ...
[main t5] python_exec("sc.pl.umap(adata, color='cell_type', save='hca_cell_type_pca.png')")
[main t6] finish(message="figure saved")
```

Parent context accumulates branch summaries. No removal mechanism — same
fold-tree problem as BC-Plus.

File change: copy `agents/fold_agent.py` → `agents/fold_agent_code.py`, swap
tool registration.

---

## 4. Agent Loop C: `code_graph_isolated_agent` (full CG)

Branch + return + graph state. Same operations as BC-Plus's
`graph_agent_isolated` but with TWO NEW node types specific to code domain:

```python
class NodeType(Enum):
    QUERY       = "query"        # original task instruction (existing)
    OBSERVATION = "observation"  # python_exec stdout (was: search result)
    ACTION      = "action"        # tool call descriptor (existing)
    SUBTASK     = "subtask"       # branch root (existing)
    SUMMARY     = "summary"       # collapsed branch (existing)
    VARIABLE    = "variable"     # NEW: data DataFrames, models, predictions
    HYPOTHESIS  = "hypothesis"   # NEW: model choice, featurizer choice
```

**VARIABLE node** content example:
```
[VARIABLE n5] name=adata, type=AnnData, shape=(20000, 33538),
              defined_at_turn=1, used_in=[t2, t3, t4, t5]
```

**HYPOTHESIS node** content example:
```
[HYPOTHESIS n8] "use PCA with n_comps=30 then UMAP"
              proposed_at_turn=3, tested_at_turn=4, status=succeeded
```

These nodes are POLICY-EMITTED (agent calls add_node for them) or
AUTO-EXTRACTED (env parses python_exec output for `var = ...` assignments
and adds VARIABLE nodes deterministically).

### Graph evolution on task #69 (UMAP):

```
turn 1: agent emits python_exec("adata = sc.read('hca/...h5ad')")
  → env runs code, captures stdout
  → env auto-adds OBSERVATION node n2 (stdout)
  → env auto-adds VARIABLE node n3 (name=adata, type=AnnData, ...)
  → env auto-adds CAUSAL edge n2 -> n3 (observation revealed this variable)
  Graph: [n1 query, n2 obs, n3 var(adata)]

turn 2: agent emits branch("Explore data schema")
  → graph.spawn_child(subtask_id=n4)
  → child branch graph rooted at "Explore data schema"
  [branch turns 1-3 happen in child graph]
  → branch returns "cell_type column found, 18432 genes survive filter"
  → graph.collapse_child(n4)
  → parent graph adds SUMMARY node n5 (branch_message)
  → IMPROVEMENT #3: auto-bind n5 → most-related existing parent node
    via lexical Jaccard. Matches "adata" mention → bind n5 → n3
    via SEMANTIC edge.
  Graph: [n1, n2, n3(adata), n4(subtask), n5(summary)]
         edges: n2→n3, n4→n5 (CAUSAL), n5→n3 (SEMANTIC auto-bind)

turn 3: agent emits add_node(content="use PCA 30 components then UMAP",
                            node_type=HYPOTHESIS, parent_id=n1)
  → graph adds HYPOTHESIS node n6
  Graph: ..., n6(hypothesis)

turn 4: agent emits python_exec("sc.tl.pca(adata, n_comps=30)")
  → env runs, adds OBSERVATION n7 + VARIABLE n8 (adata.obsm['X_pca'])
  → auto-bind n8 → n3 (lexical match on "adata") via SEMANTIC

turn 5: agent emits python_exec("sc.pl.umap(...save='pred_results/hca_cell_type_pca.png')")
  → env runs, adds OBSERVATION n9 (figure saved confirmation)

turn 6: agent emits finish(message="saved hca_cell_type_pca.png")
  → graph.compute_graph_reward()
  → uniqueness_bonus: 1 summary node → 0 (need ≥2)
  → graph_n_edges = 5, graph_n_nodes = 9 (after pruning intermediate)
```

### Why VARIABLE / HYPOTHESIS nodes matter (paper claim)

In code trajectories, **the same variable is referenced N times** across
turns (`adata` is used in turn 1, 3, 4, 5). BC-Plus's entity-compression
claim ("graph wins on entity-heavy tasks") applies directly:

- **BC-Plus**: O(20+5N) tokens to track entity referenced N times (20 token
  description + 5 token reference each turn).
- **ScienceAgentBench**: SAME structure. VARIABLE node tracks adata once;
  subsequent turns reference n3 = "adata".

This is even more pronounced than BC-Plus because:
1. Code variables are DEFINITELY shared across turns (Python state persists).
2. Hypotheses (model choice) inform multiple later turns deterministically.
3. Auto-bind edges from new code outputs to prior variables come "for free"
   — same Jaccard logic on variable names.

---

## 5. Sandbox Specification

`envs/scienceagent_sandbox.py` (NEW):

```python
class CodeSandbox:
    """Stateful Python subprocess. State persists across exec calls within
    one trajectory; fresh process per trajectory."""

    def __init__(self, workdir: str, time_limit_per_call: int = 60):
        self.workdir = workdir              # task input files + pred_results/
        self.proc = subprocess.Popen(
            ["python", "-u", "-i"],         # interactive, unbuffered
            stdin=PIPE, stdout=PIPE, stderr=PIPE,
            cwd=workdir,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )

    def execute(self, code: str) -> dict:
        # Send code + sentinel marker
        self.proc.stdin.write(code + "\nprint('__SAB_DONE__')\n")
        # Collect output until sentinel
        out, err = collect_until(marker="__SAB_DONE__",
                                 timeout=self.time_limit_per_call)
        return {"stdout": out[:2048], "stderr": err[:1024], "success": "Error" not in err}

    def close(self):
        self.proc.terminate()
```

**Pre-installed conda env** `sab` (built one-time): scanpy, anndata, deepchem,
matminer, rdkit, geopandas, rasterio, neurokit2, sklearn, torch, statsmodels,
matplotlib, seaborn, xgboost, shap (full list derived from static analysis of
all 102 gold_programs in Phase D1).

---

## 6. File Diff Plan (from BC-Plus)

| New file | Source (BC-Plus) | Change |
|----------|------------------|--------|
| `agents/tool_spec_code.py` | `tool_spec.py` | Add `python_exec` tool spec, keep `branch`/`merge`/`add_edge`/`prune`/`select`/`return`/`finish` |
| `agents/prompts_code.py` | `prompts.py` | Adapt system prompts (`SEARCH_SYSTEM_PROMPT_*` → `CODE_SYSTEM_PROMPT_*`), example trajectory in prompt swapped to a UMAP example |
| `agents/react_agent_code.py` | `react_agent.py` | Swap tool dispatch search → python_exec; reuse rest |
| `agents/fold_agent_code.py` | `fold_agent.py` | Same swap |
| `agents/graph_agent_code_isolated.py` | `graph_agent_isolated.py` | Same swap + VARIABLE/HYPOTHESIS node types + env-side auto-extract VARIABLE from python_exec stdout |
| `agents/context_graph_code.py` | `context_graph.py` | Add `_extract_variable_from_exec(stdout)` helper; rest reused |
| `envs/scienceagent_sandbox.py` | (new) | Subprocess-based Python sandbox |
| `envs/scienceagent_eval.py` | (new) | Wrap `calculate_metrics.py` from upstream |
| `scripts/eval_sab_react_30b_5node.sh` | `eval_bc_baseline_30b_instruct_8node_zeroshot.sh` | Different topology (5-node), different agent loop |
| `scripts/eval_sab_fold_30b_5node.sh` | `eval_bc_foldagent_30b_instruct_8node_zeroshot.sh` | Same |
| `scripts/eval_sab_ctxgraph_30b_5node.sh` | `eval_bc_ctxgraph_30b_instruct_8node_zeroshot.sh` | Same |

---

## 7. What changes in `compute_graph_reward()` for code domain

Mostly nothing. Existing reward components transfer:
- `task_reward` ← ScienceAgentBench `calculate_metrics.py` output (binary or 0–1 score per task)
- `graph_shaping` ← unchanged (compactness, structural, merge_bonus, prune_bonus, usage_bonus)
- `uniqueness_bonus` ← unchanged (Jaccard across SUMMARY nodes)
- `cost_penalty` ← `lambda_cost × explicit_op_count` unchanged

**Since we're doing zero-shot only**, reward shape isn't being optimized
against. It's just logged for analysis. The numerator/denominator of paper
claim is `val/task_reward` averaged across 102 tasks per method.

---

## 8. Engineering Breakdown (calendar days)

| Phase | Task | Days | Output |
|-------|------|------|--------|
| **D1** | Static-scan all 102 gold_programs, build `requirements_sab.txt`, build conda env | 1 | env on Vista |
| **D2-a** | Write `tool_spec_code.py` + `prompts_code.py` (3 system prompts) | 1 | text files |
| **D2-b** | Write `react_agent_code.py` + `envs/scienceagent_sandbox.py`, hand-test on 1 task | 1.5 | first working agent |
| **D2-c** | Write `envs/scienceagent_eval.py`, run 5 tasks end-to-end with ReAct + Qwen3-8B-Instruct | 1 | first metric numbers |
| **D2-d** | Write `fold_agent_code.py`, smoke 3 tasks | 1.5 | fold working |
| **D2-e** | Write `graph_agent_code_isolated.py` + `context_graph_code.py` (VARIABLE/HYPOTHESIS), smoke 3 tasks | 2.5 | CG working |
| **D3** | Full eval: 3 methods × 102 tasks × Qwen3-30B-Instruct (and optionally Qwen3-8B) on 5-node × 4h | 1 | numbers |
| **D4** | Analysis + paper §6 write-up | 1 | tables + draft text |
| | **Total** | **~10 working days = 2 weeks** | |

---

## 9. Open Questions (need to resolve before Phase D2-a)

1. **Sandbox time limit per python_exec call**: 60s reasonable? Deep learning
   training tasks might need 300s. → propose: 60s default, 300s for
   `train_*` task IDs.

2. **VARIABLE auto-extraction**: who decides what's a "variable"? Options:
   - (a) Parse `name = ...` assignments from python_exec code → simple
   - (b) Track `dir()` diff before/after each exec → robust but slow
   - (c) Agent emits add_node(VARIABLE, ...) explicitly → most paper-honest
   - Propose: (c) for headline, (a) as "automatic baseline" ablation.

3. **HYPOTHESIS node trigger**: same options as above. Propose (c) only —
   agent must emit explicitly. Otherwise we're auto-extracting LLM reasoning
   into graph (paper-questionable).

4. **Branch sub-task examples in prompt**: should we hint at standard sub-task
   structure (Explore / Featurize / Train / Predict / Save)? Risk: too
   prescriptive, agent doesn't learn to decompose. → propose: 1 generic
   example in prompt, no domain-specific guidance.

5. **GPT-4o judge** (used by ScienceAgentBench for visualization scoring): are
   we OK with that cost (~$1-2 per task on 102 task * 3 method = $300-600)?
   Or self-host Qwen3-32B-judge? → propose: use their default GPT-4o judge to
   stay comparable to their reported numbers.

---

## 10. Success Criteria for Phase D2 (engineering done, ready for D3 full eval)

- [ ] All 3 agent loops can complete at least 3 of 5 sanity tasks (loaded data, ran code, produced output file)
- [ ] Sandbox sandbox isolation OK: trajectory 1 state doesn't leak into trajectory 2
- [ ] `[BRANCH AUTO-BIND]` log line fires at least once during a smoke run (verifies Improvement #3 transfers)
- [ ] `uniqueness_raw > 0` on at least 30% of trajectories with ≥2 branches (verifies Improvement #1 transfers)
- [ ] No `OutOfMemoryError`, no Python sandbox deadlocks, no harness timeouts

If all 5 pass → proceed to D3 (full 102 × 3 = 306 trajectory eval).

If any fail → diagnose + fix before burning quota on full eval.
