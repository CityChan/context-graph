# MemoBrain and A-MEM: BC-P zero-shot adapters

These are independent, source-guided implementations of the public memory
mechanisms, evaluated with our Qwen3.5-9B actor, local BC-P tools and judge.
They are not results from the authors' trained checkpoints or original benchmark
stacks. No RL training or cross-task memory is enabled. Existing ContextGraph
and FoldAgent defaults are unchanged.

| Method | Pinned official source | Implemented mechanism |
| --- | --- | --- |
| MemoBrain | [qhjqhj00/MemoBrain](https://github.com/qhjqhj00/MemoBrain/tree/82f16e17c28313a57bf95d83340142b96507f3d1), `src/memobrain.py`, `problem_tree.py`, `prompts.py` | Every observed episode creates subtask/evidence nodes and dependency edges. Periodic recall flushes invalid steps to their notes and folds completed sub-trajectories into summaries, rewiring edges. The next actor request uses the projected history. |
| A-MEM | [agiresearch/A-mem](https://github.com/agiresearch/A-mem/tree/ceffb860f0712bbae97b184d440df62bc910ca8d), `agentic_memory/memory_system.py`, `retrievers.py` | Analyze an observed episode into keywords/context/tags, find semantic nearest notes, generate links and evolve neighbor metadata. Retrieve nearest notes and their linked notes into subsequent actor inputs. |

Adaptations are explicit:

- Both use the same model endpoint and decoding as the actor, with a separate
  memory-role prompt. This is a zero-shot mechanism comparison. MemoBrain's
  published trained memory checkpoints are not used. Calls are sequential for
  deterministic state updates; this does not reproduce the paper's asynchronous
  co-pilot scheduling or its latency claims.
- MemoBrain recalls every five completed tool interactions, or when the current
  history reaches 16,384 tokens. It protects the first and latest two complete
  episodes and the original task (upstream protects individual message indices).
  Invalid patches are atomic and retain raw observations; protected or cyclic
  graph operations are rejected and recorded. Active summaries can be folded
  again. Flush substitutes node notes, rather than silently deleting evidence.
- A-MEM explicitly calls content analysis before adding a note. Each note stores
  the observed action/reasoning and visible tool output. Its original content
  remains immutable; evolution updates context/tags and links using stable IDs,
  not neighbor positions interpreted as global insertion indices.
- A-MEM uses the upstream default embedding family, CPU MiniLM-L6-v2, pinned to
  `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. An in-memory exact cosine index
  replaces Chroma's shared collection/ANN implementation for task isolation.
  Embeddings index original note content, as in the inspected upstream code.
  Retrieval uses the task plus latest actor action, top five notes and their
  links, with a 4,096-token memory view and the latest two raw episodes. Notes
  that do not fit are omitted from the view and counted; stored notes are intact.
  This is an episode-local BC-P adaptation of a longer-term memory system.
- Malformed memory JSON never becomes a tool action. Failed MemoBrain updates
  keep the raw episode; failed A-MEM analysis retains a raw note without inferred
  metadata. Format failures are counted and fail the strict smoke audit. Transport,
  embedding and judge failures propagate as evaluation errors, not zero rewards.
- The v2 adaptation uses the shared `search_single_v1` actor prompt, also used by
  SUPO (AgentFold v3 has its own state-specific prompt). It removes the inherited parallel-search example that
  conflicted with the single-call parser. A rejected actor reply is shown as a
  bounded excerpt on retry, followed by a correction; no rejected tool executes.
  Three consecutive failures stop as `invalid_tool_limit` and fail the evaluation
  summary audit. A valid call resets this counter. Memory-JSON validation remains
  separate. The prompt and retry policy are recorded in `baseline_protocol`.

## Budget and evidence

The shared evaluator keeps the existing BC-P protocol: 32,768-token context,
24,576 cumulative generated/visible-observation/control-instruction tokens,
100 model requests, 2,048 tokens per request, 1,024 final-answer reserve,
and unchanged search/open-page limits. Memory replies are capped at 1,024 tokens
and count against **both** the cumulative budget and model-request limit.
Removing context never refunds consumed tokens. These methods therefore do not
receive 100 actor requests plus unlimited helper requests.

Replayed inputs are reported in `total_token` (input plus output across every
request), separate from the cumulative response/observation budget. Embedding
calls and elapsed time are reported too. This controls the stated budgets but
does not assert equal FLOPs or wall-clock cost between methods.

Every task saves exact actor/helper inputs and outputs in `requests-*.jsonl`
and `trajectory-*.json`; the trajectory also contains `memory_audit`, final
`memory_state`, and `baseline_protocol`. No reference answers or judge feedback
are passed into memory construction. State resets for every task.

## Four-node Vista idev smoke

From the existing allocation and repository checkout:

```bash
conda activate cxtgraph
git pull --ff-only
bash scripts/smoke_bcp_graph_memory_qwen35_9b_4node_idev.sh both
```

This runs three fixed-seed questions per method, using one retrieval node and
three model/evaluator nodes, sequentially for the two methods. The same selected
questions and actor checkpoint are used. Each launch releases its own services
before the next one starts; allocation time still applies. To use two separate
four-node allocations, run one of these in each:

```bash
bash scripts/smoke_bcp_graph_memory_qwen35_9b_4node_idev.sh memobrain
bash scripts/smoke_bcp_graph_memory_qwen35_9b_4node_idev.sh amem
```

The script uses `cxtgraph` for evaluation and `deepseek_v4` for serving. A-MEM
requires `sentence_transformers` in the existing agent environment (already a
retrieval-stack dependency); the script does not install into or replace it.
It downloads only the pinned MiniLM files to `$SCRATCH/hf_cache` if needed and
checks actual embedding inference. If compute-node internet is unavailable,
prepare that cache once on the login node using the same environment:

```bash
HF_HOME="$SCRATCH/hf_cache" HF_HUB_CACHE="$SCRATCH/hf_cache/hub" python scripts/prepare_amem_embedding.py
```

Outputs are under the printed `$SCRATCH/bcp-graph-memory-JOBID-XXXXXX/` directory:
`memobrain/` and `amem/` each contain `summary.json` and
`memory-smoke-audit.json`. The strict audit requires complete shards, clean
execution/judging/generation, exact request accounting, successful memory
construction, and no invalid memory decisions. It reports fold/flush/link counts;
a model may legitimately choose no graph edits on a short task. Smoke success
does not require a correct answer and does not establish benchmark performance.

For eight questions, use `SAMPLES=8` with the same script. All substantive
configuration is saved in the manifests. Supported overrides are
`MEMORY_HELPER_MAX_TOKENS`, `MEMOBRAIN_RECALL_INTERVAL`,
`MEMOBRAIN_CONTEXT_THRESHOLD`, `AMEM_TOPK`, and `AMEM_MEMORY_TOKENS`.

## Local regression

```bash
python -m pytest tests/test_graph_memory_baselines.py tests/test_agentfold.py tests/test_supo.py tests/test_bcp_qwen38_eval.py tests/test_emergency_finalizer.py tests/test_audit_bcp_pair.py -q
```

Mocked-model tests validate operations, input projection, budgets, failure
isolation and dispatch. The pinned MiniLM probe validates real CPU embeddings.
Neither is a substitute for a Vista Qwen GPU smoke run.

The initial implementation check passed 111 focused tests plus 25 generation-audit
tests. One additional ScienceWorld audit test has a pre-existing fixture missing
`memory_profile`; the identical failure was reproduced on the untouched parent
commit `5e2487c`. This change does not modify that test or ScienceWorld runner.
