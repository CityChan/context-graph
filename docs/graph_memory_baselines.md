# MemoBrain and A-MEM: search and native-task zero-shot adapters

These are independent, source-guided implementations of the public memory
mechanisms, using our Qwen3.5-9B actor. BC-P uses local retrieval and its judge;
the native adapters use each environment's tools and grading (see below).
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
- The v3 adaptation fixes two observed control failures: memory helpers spending
  all 1,024 tokens on reasoning before emitting JSON, and final requests continuing
  to search despite the exhausted research budget. Memory helpers now render with
  thinking disabled and use phase-specific JSON schemas (including subtask/evidence
  node kinds). They still pass the existing atomic graph/neighbor validators;
  invalid semantic edits remain failures, and empty valid patches remain allowed.
  There are no added helper retries or extra token/turn allowances. Concise helper
  instructions discourage copied example placeholders; schema validity does not
  establish summary faithfulness.
- Budget-ending actor requests use a separate finish-only system prompt, thinking
  disabled, and bounded finish XML (at most 384 answer characters, further reduced
  for small remaining budgets). A character bound is not a token guarantee. Local
  validation still rejects a server response that violates the contract, with the
  same three-consecutive-error limit. Final retries now retain the specific error
  feedback. Normal research decoding and memory operations are unchanged. The new
  instructions and every rejected response are charged to the cumulative budget;
  manifests record `zero_shot_bcp_adaptation_v3`, `memory_json_schema_v1`, and the
  final-output profile. Re-run smoke before treating this revision as GPU-validated.

## Budget and evidence

New evaluations use `token_budget_mode=working_context`: the 32K/64K setting
limits each request's fully rendered input plus output. There is no episode
token cap inferred from the window. Compression can recover context space;
past generation, observations and helper work remain recorded. All actor and
helper calls still share the model-request limit (normally 100), with unchanged
2,048-token actor and 1,024-token helper caps. BC-P keeps its 1,024-token final
reserve. SWE now disables forced finalization for every method and grades the
workspace patch at termination. Simulators retain environment-owned termination.

`AGENT_BUDGET_MODE=legacy_cumulative` explicitly restores the old evaluator
budget/finalization configuration for reproduction. Historical run descriptions
below refer to that old protocol. See [evaluation budgets](evaluation_budgets.md)
for the comparison boundary and rerun commands.

Replayed inputs are reported in `total_token` (input plus output across every
request), separate from the cumulative response/observation ledger. Embedding
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

The script uses `cxtgraph` as the base agent environment and `deepseek_v4` for
serving. A-MEM requires `sentence_transformers`; some `cxtgraph` installations
do not contain it. The smoke wrapper now detects this and creates a unique
`--system-site-packages` venv under `$SCRATCH/context-graph-agent-benchmarks/envs`.
It installs `sentence-transformers==5.3.0` and missing dependencies there, while
constraining every existing base distribution to its installed version. A version
conflict fails setup instead of upgrading the existing Torch/Transformers stack.
Search, serving, and MemoBrain still use their existing interpreters. The selected
A-MEM interpreter is saved in `amem-python.txt` and forwarded to every evaluator;
each evaluator probes the pinned embedding offline before starting its tasks.
Existing installations are import-tested and reused without pip installation.

A-MEM preparation and evaluator processes explicitly select PyTorch before
importing Transformers (`USE_TF=0`, `USE_TORCH=1`, `USE_FLAX=0`,
`FORCE_TF_AVAILABLE=0`). This avoids importing an inherited incompatible Keras 3
installation; installing `tf-keras` is unnecessary for this CPU embedding path.
These settings do not modify the installed TensorFlow/Keras packages. If setup
installed dependencies but failed its import probe, reuse that venv by setting
`AMEM_AGENT_PYTHON` to its `bin/python` on the next smoke invocation.

The wrapper downloads pinned MiniLM files to `$SCRATCH/hf_cache` if needed and
checks actual embedding inference and semantic retrieval. If compute-node internet
is unavailable, prepare the environment and cache once on the login node:

```bash
python scripts/prepare_amem_environment.py --root "$SCRATCH/context-graph-agent-benchmarks/envs" --python-file "$SCRATCH/amem-python.txt"
export AMEM_AGENT_PYTHON=$(cat "$SCRATCH/amem-python.txt")
env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE HF_HOME="$SCRATCH/hf_cache" HF_HUB_CACHE="$SCRATCH/hf_cache/hub" "$AMEM_AGENT_PYTHON" scripts/prepare_amem_embedding.py
```

Export the printed interpreter as `AMEM_AGENT_PYTHON` in the idev shell to reuse
it; otherwise the wrapper provisions its own environment. This setup path does
not change the embedding model, memory mechanism, or evaluation budget.

Outputs are under the printed `$SCRATCH/bcp-graph-memory-JOBID-XXXXXX/` directory:
`memobrain/` and `amem/` each contain `summary.json` and
`memory-smoke-audit.json`. The strict audit requires complete shards, clean
execution/judging/generation, exact request accounting, successful memory
construction, and no invalid memory decisions. It reports fold/flush/link counts;
a model may legitimately choose no graph edits on a short task. Smoke success
does not require a correct answer and does not establish benchmark performance.

The wrapper also runs the audit when evaluation exits with a failure. Its `tasks`
section shows each termination reason, actor/helper request counts, budget use,
the last three rejected actor responses (bounded tails and output-limit flags),
and memory-update errors. Inspect an existing completed run without starting any
services with `python scripts/audit_graph_memory_smoke.py /path/to/memobrain`.
The audit keeps its nonzero exit status when checks fail; diagnostics do not
relax the pass criteria or change the saved scores.

For eight questions, use `SAMPLES=8` with the same script. All substantive
configuration is saved in the manifests. Supported overrides are
`MEMORY_HELPER_MAX_TOKENS`, `MEMOBRAIN_RECALL_INTERVAL`,
`MEMOBRAIN_CONTEXT_THRESHOLD`, `AMEM_TOPK`, and `AMEM_MEMORY_TOKENS`.

## Local regression

MemoBrain `zero_shot_bcp_adaptation_v5` uses recall target profile
`active_unprotected_sorted_ids_regex_v2`. Each recall builds a JSON-output regex
from the same graph snapshot shown to the helper: only active, unprotected
node IDs are selectable, and IDs within each fold must be strictly increasing.
Every nonempty subset remains available; single-node folds and multiple
fold/flush operations remain supported. With no editable nodes, both operation
lists must be empty. JSON keys follow the prompt's example order. Other helpers
retain their JSON schemas.

The v4 live smoke finished all three tasks with five folds, but one request
emitted duplicate IDs `[3,4,4]`. ID enums alone do not prevent duplicates.
The regex enforces uniqueness within a fold without relying on `uniqueItems`,
which the [vLLM XGrammar backend lists as unsupported](https://docs.vllm.ai/en/latest/api/vllm/v1/structured_output/backend_xgrammar/).
It uses neither lookarounds nor backreferences and grows quadratically with the
editable-node count, rather than enumerating all subsets.

The model still selects operations and writes notes; there is no post-generation
deduplication or repair. Cross-operation overlap, graph cycles, and existing
transactional checks remain enforced; any rejection still fails the strict
smoke audit. The request budget is unchanged and the added instructions are
charged. Local tests do not establish live decoder behavior or task accuracy;
rerun the same smoke in a fresh output directory.

A-MEM `zero_shot_bcp_adaptation_v4` introduced the evolution profile
`neighbor_aligned_json_v1`. The v3 live smoke finished 3/3 tasks and applied six
evolutions, but reported ten invalid memory requests; the displayed errors were
`Evolution arrays must align with neighbor IDs`. The output schema now uses the
actual ordered neighbor snapshot, with separate branches for no change,
strengthening links, updating neighbors, or both actions. `update_neighbor`
requires exactly one context string and one tag list per supplied neighbor,
including unchanged neighbors. `suggested_connections` only selects links from
that neighbor set; it does not determine the update-array length. Inactive
action fields are empty, and no change uses `should_evolve=false` with empty
arrays. The prompt repeats the ordered IDs and required count.

The model still chooses actions, links, and metadata. Array counts do not prove
that summaries are faithful or assigned to the correct neighbor. There is no
automatic padding, broadcasting, truncation, extra retry, or relaxation of the
atomic update validator or strict smoke audit. Existing token and request
limits still apply; the new instructions are charged. Local schema/runner
tests must be followed by a fresh Vista smoke before claiming integration pass.

```bash
python -m pytest tests/test_graph_memory_baselines.py tests/test_agentfold.py tests/test_supo.py tests/test_bcp_qwen38_eval.py tests/test_emergency_finalizer.py tests/test_audit_bcp_pair.py -q
```

Mocked-model tests validate operations, input projection, budgets, failure
isolation and dispatch. The pinned MiniLM probe validates real CPU embeddings.
Neither is a substitute for a Vista Qwen GPU smoke run.

## A-MEM bounded metadata and current live evidence

The supplied MemoBrain run at `f73edd0` passed the strict BC-P smoke: three
finished tasks, 26 valid memory requests, six folds, no invalid decisions. It
scored 0/3; this is evidence of mechanics, not accuracy or flush coverage.
The supplied A-MEM run at `9f8e472` finished all three tasks with 15 evolutions,
but one of 35 memory requests repeated tags until its 1024-token output limit.
It remains a failed smoke. The previous neighbor-array errors were absent in
that run; the remaining JSON was truncated, not an embedding failure.

A-MEM `zero_shot_bcp_adaptation_v5` adds `bounded_metadata_v1`: each context is
limited to 240 characters, each label to 32 characters, each tag list to four
items, and analysis keywords to eight items. Both analysis and evolution
schemas carry these limits. Raw note content remains unchanged. These are
character/item bounds, not token guarantees; all rejected decisions still fail
the strict audit. No output is silently repaired and no budget is increased.

The subsequent supplied Vista run at `29d2e6f`,
`bcp-amem-smoke-1062642-wvtiIR/amem`, passed the strict BC-P smoke: three
finished tasks, 32/32 valid memory updates, 14 evolutions and 55 embedding calls.
Task accuracy remained 0/3. Its following SWE smoke stopped before evaluation:
the newer Hub client rejected the offline full-repository snapshot because the
preparation step intentionally downloaded only PyTorch assets. Preparation and
runtime now share `embedding_snapshot()`, including the same JSON/TXT/safetensors
filter and pinned revision. The runtime stays offline; missing required assets
still fail, and unused ONNX/OpenVINO exports are not downloaded. This cache fix
has local regression coverage and passed the supplied native SWE A-MEM run at
`f670711` (details below).

## Native SWE-bench Lite, ScienceWorld and DiscoveryWorld

Both methods have native adapters and four-node launchers. A-MEM has a supplied
live SWE smoke pass; the remaining native results are described below.
Use `scripts/smoke_stateful_memory_qwen35_9b_4node_idev.sh` with
`memobrain` or `amem` and one of `swe-lite`, `scienceworld`, `discoveryworld`.

SWE retains the existing ARM-compatible instance subset, sandbox, patch export
and grading environments. Actor calls are `python_exec(code)` and
`finish(message)`. ScienceWorld uses `action(command)` and `focus_v3`;
DiscoveryWorld uses JSON actions, `discoveryworld_v1` and `compact_v1` in the
smoke launcher. Both simulators determine termination and keep native scores,
including failures. Neither exposes a finish tool. Private scorecards and gold
solutions do not enter memory prompts.

Memory helpers receive dispatched calls and shown observations, with explicit
instructions to preserve object IDs, file changes, failures and unresolved
goals. Recent observations supersede older notes. Memory never resets, replays
or forks the environment. All actor and helper requests count toward the model
turn cap; a 100-request cap therefore does not promise 100 environment actions.
Generated tokens, shown observations and control instructions share the
cumulative budget without refunds. Simulators do not reserve a final-answer
slot. SWE retains the bounded finish call. Per-task trajectories save the
memory state, request audit, actual method settings and native protocol.

A-MEM setup starts from the benchmark's own agent/simulator Python and, if
needed, creates a private sentence-transformers overlay inheriting that
environment's package paths. It probes pinned MiniLM embeddings before starting
model services. The BC-P `AMEM_AGENT_PYTHON` override is not reused for native
tasks; native setup selects `AMEM_NATIVE_PYTHON` and records its interpreter and
package versions in the run directory. SWE preparation/grading keep their
original interpreters. Existing environments and running jobs are preserved.

After the BC-P smoke passes, run the following in one four-node idev allocation
for MemoBrain, and the second command in the other allocation for A-MEM:

```bash
git pull --ff-only
for benchmark in swe-lite scienceworld discoveryworld; do SAMPLES=2 bash scripts/smoke_stateful_memory_qwen35_9b_4node_idev.sh "$benchmark" memobrain || break; done
```

```bash
git pull --ff-only
for benchmark in swe-lite scienceworld discoveryworld; do SAMPLES=2 bash scripts/smoke_stateful_memory_qwen35_9b_4node_idev.sh "$benchmark" amem || break; done
```

Each command creates a fresh artifact directory. The native audit requires
completed grading, native provenance, exercised tools, successful memory
construction and at least one MemoBrain fold or A-MEM evolution across the
selected tasks. It also checks memory errors, actor/helper request accounting,
token accounting and A-MEM semantic retrieval. A short terminal failure that
never exercises memory cannot alone establish a pass. Accuracy is reported by
native grading, not used as the integration gate. Full experiments should wait
for the corresponding native smoke evidence.

### A-MEM native smoke evidence and audit directory filtering

The supplied `f670711` run `swe-lite-amem-1062642-V2B4sq` passed: both tasks
were graded and finished, with 106 valid memory updates, 52 evolutions and no
memory errors. Two actor output-format errors recovered. This establishes
integration mechanics, not patch correctness.

The following `scienceworld-amem-1062642-gFEk76` run graded both tasks, each
with 34 actor requests and 66 valid memory requests, reaching the 100-request
cap. Both scores remained zero. Every audit check except `complete` passed.
The auditor incorrectly counted `amem-native-python.txt` and
`amem-native-environment.txt` as shards because they matched `amem-*`.
Shard discovery now counts directories only. Missing summaries, missing results,
pending tasks and extra shard directories still fail the completeness check.
The `max_turn` stops and zero scores are preserved; neither implies missing
evaluation output. Regression fixtures cover this layout for both simulators.

The supplied re-audit after `602573e` passed without rerunning ScienceWorld.
The subsequent `discoveryworld-amem-1062642-UN4tIb` smoke also passed: two
graded tasks, 66 valid memory updates, 32 evolutions and no memory errors.
Both tasks exhausted the cumulative token budget with zero scores; 17 invalid
environment actions and one actor format error remain in the artifacts.
A-MEM therefore has supplied mechanics passes on BC-P and all three native
benchmarks, without a task-performance claim.

### Native MemoBrain recall v2

The supplied SWE smoke at `29d2e6f` graded both tasks and exercised five folds
and fourteen flushes, but rejected eight memory decisions. The displayed recall
errors repeated flush targets and reused targets in both flush and fold. The
v1 per-fold ID ordering did not guarantee disjoint operations across a request.
The five displayed actor format errors recovered; they were not the failing
smoke gate. ScienceWorld and DiscoveryWorld were not reached by the shell loop.

Native MemoBrain at `73301fe` recorded `zero_shot_native_adaptation_v2` with
`recall_target_profile=node_assignment_regex_v1`. The recall wire format is
`{"nodes":{"3":null,"4":{"flush":"reason"},"5":{"fold":1,"rationale":"reason","notes":[{"role":"assistant","content":"observed findings"}]}}}`.
Keys are exactly the sorted active, unprotected node IDs, each appearing once.
Each value chooses keep, flush, or membership in a model-selected fold group.
Equal positive integer group labels share one fold, including interleaved node
groups; labels are not node IDs. An empty nodes object is a valid no-op.

The adapter translates these assignments to the unchanged `maintain()` API.
Within each group, model-written notes are concatenated in node order, with
no host-written summary or automatic target repair. Both the raw assignment and
translated operations are audited. The model still selects every target and
group and writes the notes; arbitrary disjoint groups and concurrent flushes
remain representable. The regex grows linearly with the editable-node count.
Raw output is validated before JSON parsing so duplicate keys cannot collapse
silently. Invalid output or graph cycles still reject the entire update.

This changes native recall serialization and the granularity of summary notes;
it is a versioned adaptation, not an identical prompt to native v1. Explicit
per-node slots and notes consume the existing 1024-token helper allowance and
can still exhaust it. No request/token budget is increased and no extra retry
is added. BC-P retains its previously validated v5 recall protocol; A-MEM is
unchanged. Local tests cover disjoint partitions, state/protection changes,
malformed outputs, atomic cycle rejection, native dispatch and budget accounting.
The supplied `swe-lite-memobrain-1062643-S5SgPE` smoke at `73301fe` finished
and graded both tasks, but failed memory-format validation: the first task had
10 invalid memory requests, the second had none. The displayed failures include
two recalls repeating the same long note for many nodes and a memorize patch
adding repeated nodes; all three outputs reached the 1024-token helper limit.

### Native MemoBrain bounded group summaries v3

`zero_shot_native_adaptation_v3` uses `group_summary_regex_v2`: each editable
node maps to `null` (keep), `"flush"`, or a group number 1 through 4. A shared
`flush_rationale` describes the flushes. The separate `folds` object contains
slots `"1"` through `"4"`; each used slot contains one model-authored rationale
and one note covering all assigned nodes, while unused slots are null. The
minimal no-op remains `{"nodes":{}}`. For example:

```json
{"nodes":{"3":1,"4":1,"5":"flush"},"flush_rationale":"superseded evidence","folds":{"1":{"rationale":"completed investigation","notes":[{"role":"assistant","content":"Observed findings, file changes and unresolved goals for nodes 3 and 4."}]},"2":null,"3":null,"4":null}}
```

Each group note is capped at 240 characters and each rationale at 64. The
model may choose any editable nodes, including interleaved groups, but may
create at most four fold groups per request. Remaining nodes stay active.
The adapter requires summaries to match assigned groups exactly, then maps
each summary once to the unchanged atomic maintenance API. It does not merge,
deduplicate, synthesize or truncate notes. Raw regex checks still reject
duplicate node keys before parsing; graph protection and cycle checks remain.

Native memorize requests use `bounded_patch_v1`: at most two new nodes and
four edges, at most two 240-character notes per node, 64-character edge
rationales and 16-character temporary IDs. Prompts ask for new evidence from
the current interaction rather than copies of existing nodes. Decoder schema
caps and runtime bounds reject oversized patches; raw episodes remain stored.

These are versioned output restrictions recorded in provenance, not a claim
of unchanged upstream prompts. Character bounds do not guarantee token fit,
especially with many editable IDs or a nearly exhausted global budget. All
outputs, including failures, retain existing token/turn charges and the
1024-token helper cap; there are no extra retries or silent repairs. BC-P and
A-MEM protocols are unchanged. Local tests cover group partitions, large
shared summaries, missing/unused summaries, truncation, patch bounds, atomic
cycle rejection, raw-episode preservation and request accounting. A fresh
native Vista smoke is required to validate this version with the live decoder.

The initial implementation check passed 111 focused tests plus 25 generation-audit
tests. One additional ScienceWorld audit test has a pre-existing fixture missing
`memory_profile`; the identical failure was reproduced on the untouched parent
commit `5e2487c`. That initial BC-P change did not modify the test or ScienceWorld runner.
