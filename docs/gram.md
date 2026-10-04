# GRAM baseline and BC-P adaptation

This is an independent, paper-guided implementation of **GRAM: Empowering Agent
with Actively Managed Graph-Structured Memory via Reinforcement Learning** (the
anonymous 16-page PDF supplied by the user). It does not change ContextGraph,
FoldAgent, their saved results, or the ScienceWorld protocol. GRAM's task here is
document-stream question answering, not a replacement ScienceWorld environment.
Source PDF SHA-256: `fc4843b1d5fa8680752e3944d4554e7eec3f0826be83cbdff1cd80e8b9846c3a`.

## BC-P zero-shot smoke on four Vista idev nodes

The BC-P entrypoint is an **adaptation**, not the paper's document-stream benchmark.
In an idle four-node allocation, run:

```bash
cd /work/09281/chc_1996/vista/context-graph && git pull --ff-only origin master && GRAM_SAMPLES=2 bash scripts/smoke_gram_bcp_qwen35_9b_4node_idev.sh
```

The script starts local BC-P corpus retrieval, a pinned Qwen3.5-9B actor, a
separate fixed Qwen3.5-9B memory helper, and an evaluator on one node each.
It reuses an exported `OPENAI_API_KEY`, or loads the existing `.openai_env`
from `$WORK`, `/work/09281/chc_1996/vista`, or `$HOME`. No key re-entry is needed.
As in the BC-P launcher, `JUDGE_BASE_URL` explicitly overrides the judge endpoint;
otherwise a stale actor `OPENAI_BASE_URL` is cleared. Credentials are not logged.
It reuses the cached corpus/embedding service in `cxtgraph` and existing vLLM
server setup. No Docker, new allocation, package installation, or training.
The default evaluator Python is the existing `agent-direct-Rs4ngP` environment;
override `GRAM_PYTHON` with another compatible interpreter if necessary.
`DATA_PATH` defaults to `data/bc_test.parquet` and must already exist. In a linked
worktree without that local file, the launcher resolves the primary checkout's
`data/bc_test.parquet` through Git's common directory. An explicit `DATA_PATH`
always takes precedence; a missing explicit path fails rather than selecting
another dataset. Missing evaluator Python and missing data have separate errors
that print the exact path. The resolved interpreter/data paths are logged.
The two rows are selected by the existing BC-P seeded sampling function (seed 42).
Use `GRAM_SAMPLES=8` for a larger pilot or `-1` for all rows; tasks run serially
and still obey the allocation's wall-clock limit.

External `<search>` / `<open_page>` reuse `LocalSearch` ranking, repeat-snippet
handling, top-k cap 5, 128-word/2000-character snippets, and
4096-word/48000-character opened pages. `<memory_search>` only searches the
stored graph. The actor must consume each returned tool observation through
Insert/Update before requesting another; provenance retains an observation ID,
with its corpus docids in the trajectory. A corpus search is required before
Answer. Reference labels never enter actor/helper requests.

Defaults: 65,536 context, 24,576 **actor output** tokens per episode, 2,048 per
step, 100 policy actions, 3,600 seconds per episode, temperature 0, thinking
disabled, exact-name entity matching plus helper canonicalization. These are
**not a cost-matched comparison** to the existing thinking-enabled
ContextGraph/FoldAgent runs. Each memory action costs a policy step and helper
inference is additional; helper usage is logged in `trajectory.jsonl`.
Override `GRAM_CONTEXT_LENGTH`, `GRAM_EPISODE_TOKENS`, `GRAM_MAX_STEPS`, or
`GRAM_SEED` explicitly and preserve the resulting manifest.

Scoring reuses BC-P strict matching followed by the configured judge
(`JUDGE_MODEL`, default `gpt-5-nano`). Missing/failed judge or retrieval calls
are infrastructure errors, not incorrect answers. This entrypoint reports
`accuracy`, not document-stream F1 or GRAM training reward.

Artifacts are under `$SCRATCH/context-graph-gram/runs/gram-bcp-9b-JOBID-XXXXXX/`:
`suite.log`, `preparation.log`, `search.log`, `server-actor.log`,
`server-memory.log`, `evaluator.log`, and `evaluation/summary.json`.
`evaluation/instances/*/attempt-*/` contains trajectories, sampled actor
requests, segments, and pre-judge episode records. The launcher records code
commit/diff and cleans up only its own Slurm steps, never the idev allocation.
CPU tests cover the mocked HTTP round trip and launcher lifecycle; a live
Vista BC-P smoke is still required to establish deployment success.

## Implemented contract

Sections 3.1–3.3 and Appendix C specify a directed entity/relation graph and one
atomic XML action per policy step:

| Action | Effect | Consume current document? |
| --- | --- | --- |
| `memory_insert` | Entity extraction, verified-entity relation extraction, add triples | Yes |
| `memory_update` | Remove explicitly invalidated triples and add replacements | Yes |
| `memory_search` | Retrieve directed multi-hop paths from the existing graph | No |
| `answer` | Terminate with an answer | No |

All graph edits are validated atomically. Edges retain source document IDs;
search results are invalidated after edits. No future document, supporting-fact
annotation, reference answer or external web search is available to the policy.
The memory helper receives the current document and question, never the answer.
Helper and actor requests are separate and audited. Only actor tokens enter RL.

Section 3.3 uses answer token F1 plus `0.1 * format_reward`. Evaluation reports
answer F1 separately: the training reward is **not** a QA accuracy metric.
Failed infrastructure is recorded separately and makes the evaluator exit 2;
an ordinary wrong or unfinished answer receives zero answer F1. Do not infer
scientific performance from a successful process exit.

## Explicit reproduction choices and limitations

The PDF does not specify the helper model, entity embedding model/threshold,
search ranking/top-k/hop limit, exact dataset split/filter/order, decoding budgets,
or how per-step format scores are combined. Defaults below are implementation
choices, not recovered paper settings:

- A separately configured frozen LLM extracts/maintains triples. It must produce
  strict JSON; helper schema failures are infrastructure errors, not policy failures.
  During RL the helper must not be the changing policy server. The launcher requires
  an explicit frozen-service acknowledgement; it cannot remotely prove immutability.
- Optional OpenAI-compatible `/v1/embeddings` supplies entity vectors. New entity
  names merge into the closest existing canonical name when cosine similarity is
  strictly above `0.9`. Without an embedding endpoint, normalized exact matching
  plus helper name canonicalization is used and recorded as an **approximation**.
  Embedding similarity alone can merge distinct entities: tune/audit on training or
  validation data, never against held-out answers.
- Search seeds use lexical entity/relation overlap, followed by directed paths
  of at most 2 edges; at most 12 paths are returned, longer paths first. Following
  the entity-index example in Table 5, the actor sees canonical names, node degrees
  and relation labels; it retrieves actual linked triples through Search. The exact
  index serialization is our choice. The full graph stays in helper input/audit.
- Each actor call gets a fresh prompt containing question, graph index, last retrieval,
  current document and last error. Old raw documents/reasoning are not replayed.
  Overflow fails explicitly rather than silently deleting evidence. Large graphs
  or full TriviaQA articles may require a larger context; no hidden truncation.
- Format reward is the **mean** of per-step XML validity indicators. One optional
  preceding `<think>` block is accepted. Qwen template-level thinking is disabled
  for predictable XML/JSON parsing; the actor may emit its own tagged reasoning.
  Native-thinking/hidden-reasoning tokens are not silently repaired.
- Inserting `None` skips an irrelevant document, as in Table 5. `No relevant facts`
  is also accepted as a no-op insert. After exhaustion, only Search and Answer have
  effects; this exhaustion behavior is an implementation choice.
- Defaults: 64 actions, 2,048 generated tokens/action, 32,768 generated actor tokens
  across the episode, 32,768 working context, 1,800 seconds. Fewer than 10 remaining
  generation tokens end the episode rather than issuing a request the client rejects. Helper tokens are
  separately logged and are additional cost. Evaluation temperature is 0; training
  temperature is 1. A wall-clock timeout is treated as an infrastructure failure.
- Appendix C's Insert/Update helper headings are reversed relative to the main
  text. We follow the **main-text semantics**, not the conflicting headings.

Appendix A.2 reports Qwen2.5-3B-Instruct and Qwen3-4B, GRPO group size 5, AdamW
LR `1e-6`, warmup ratio `0.285`, `low_var_kl` coefficient `0.001`, minibatch 8,
gradient checkpointing and optimizer CPU offload. The training wrapper sets those
algorithmic parameters. Its vLLM/FSDP runtime, default 50 updates, hardware topology
and any Qwen3.5-9B run are **extensions**, not the paper's four-3090 SGLang setup.
No trained GRAM checkpoint or reproduced paper score is included.

## Prepare local datasets

Use the official releases and keep their source split provenance. Adapters accept
[HotpotQA](https://github.com/hotpotqa/hotpot),
[2WikiMultihopQA](https://github.com/Alab-NII/2wikimultihop),
[MuSiQue](https://github.com/StonyBrookNLP/musique) answerable JSONL, and native
[TriviaQA](https://github.com/mandarjoshi90/triviaqa) `Data` JSON plus its evidence
directory. Native TriviaQA needs `--evidence-root /path/to/evidence`, containing
`wikipedia/` and `web/`. This is not an adapter for every Hugging Face schema.

```bash
python -m scripts.prepare_gram_data --source /path/to/hotpot_dev_distractor_v1.json --benchmark hotpotqa --split validation --output outputs/gram-data/hotpot-validation --parquet
```

The output contains `tasks.json` (public), `references.json` (grading only), optional
`data.parquet` (VERL), and a checksummed `manifest.json`. Existing data directories
are immutable. Documents retain their input order and distractors; no support-label
selection or answer-dependent filtering occurs. Missing/unanswerable labels fail
explicitly. The paper's reported row counts are not asserted to match these files.

For small local smoke data, `--benchmark canonical` accepts rows of
`{"task_id": "...", "question": "...", "documents": [{"id": "0", "title": "...", "text": "..."}], "answers": ["..."]}`.

## Evaluate an already served model

For a **four-node Vista idev zero-shot smoke**, run this from the allocation:

```bash
cd /work/09281/chc_1996/vista/context-graph && git pull --ff-only origin master && bash scripts/smoke_gram_qwen35_9b_4node_idev.sh
```

The script uses two server/evaluator pairs, the pinned Qwen3.5-9B checkpoint,
64K context, and two bundled real HotpotQA validation questions (one per pair).
Actor and memory helper share each pair's fixed server. No embedding service is
started; this smoke exercises the documented exact-name/canonicalization variant.
No model training, dataset download, or environment installation is performed.
`GRAM_PYTHON` defaults to the existing SWE agent environment; override it with an
equivalent evaluator Python if that path is unavailable. It refuses occupied model
nodes and terminates only its own launched job steps on exit, leaving the idev
allocation intact. Its default output is printed as `artifacts=...` under
`$SCRATCH/context-graph-gram/runs/gram-smoke-9b-JOBID-...`.

`suite.log` captures the launcher, `preparation.log` captures preflight, and
`server-{0,1}.log` / `evaluator-{0,1}.log` capture the two pairs. Results and
trajectories are under `pair-{0,1}/`; each `summary.json` should report one
completed/graded row with zero infrastructure errors. Answer F1 may still be zero:
successful smoke execution does not establish benchmark performance.

Run from the repository root with its evaluation dependencies installed. The actor
endpoint must support vLLM token-ID completions; the helper uses chat completions.
For inference, both endpoints may point to the same fixed model, but total helper
inference cost must still be counted. No Docker is needed for this executor.

```bash
python -m scripts.eval_gram --data outputs/gram-data/hotpot-validation --output outputs/gram-smoke --endpoint http://ACTOR_NODE:18000 --model Qwen/Qwen3.5-9B --model-path /path/to/local/checkpoint --model-revision CHECKPOINT_SHA --memory-endpoint http://MEMORY_NODE:18000 --memory-model Qwen/Qwen3.5-9B --memory-revision CHECKPOINT_SHA --samples 2
```

For semantic entity aggregation, append `--embedding-endpoint http://EMBED_NODE:8000
--embedding-model SERVED_EMBEDDING_NAME --embedding-revision EMBEDDING_SHA` (all on
the same command line). Endpoint paths are base URLs without `/v1`. The memory
service optionally accepts `GRAM_MEMORY_API_KEY`; it is not forwarded to the
embedding endpoint or written to artifacts. Embeddings currently target trusted
unauthenticated internal services.

Use `--samples -1` for all rows; independent workers can use `--shard-count N
--shard-index I` with **different output directories**. A matching output directory
resumes finished rows. `--retry-errors` retries infrastructure failures in a new
attempt directory. Source/data/protocol changes reject resume. Model revision
strings are declared identities; server metadata is captured separately and does
not cryptographically attest the weights. Preserve endpoints' checkpoint setup.

Artifacts: `summary.json`, `manifest.json`, `observed_servers.json`, and
`instances/<id-hash>/result.json`. Each attempt has `trajectory.jsonl` (states,
actions, edits, helper messages/usage), `actor_requests.jsonl` (token IDs and API
usage), and `segments.json` (exact actor sequences). Eval log probabilities are
placeholders from the evaluation client; **do not train from those files**.

## Train inside an existing allocation / Ray cluster

### Four-node BC-P RL smoke

For **GRAM reinforcement learning on BrowseComp-Plus**, use
`scripts/smoke_gram_bcp_rl_qwen35_9b_4node_idev.sh` inside an idle four-node
idev allocation. This entry trains on BC-P, then validates on BC-P; it does not
use the bundled HotpotQA questions. Start from an isolated checkout:

```bash
cd /work/09281/chc_1996/vista/context-graph && git fetch origin && GRAM_BCP_CODE=$(mktemp -d "$SCRATCH/gram-bcp-rl-code-XXXXXX") && git worktree add --detach "$GRAM_BCP_CODE" origin/master && PROJECT_ROOT="$GRAM_BCP_CODE" bash "$GRAM_BCP_CODE/scripts/smoke_gram_bcp_rl_qwen35_9b_4node_idev.sh"
```

Topology: node 0 runs the BC-P retriever in `cxtgraph`, node 1 serves a frozen
Qwen3.5-9B memory helper, and nodes 2--3 train the actor using the existing
Qwen3.5 training environment (`deepseek_v4`). No environment is installed or
replaced. It discovers `data/bc_train.parquet` and `data/bc_test.parquet` from
the primary checkout, including when launched from a linked worktree. Override
`GRAM_BCP_TRAIN_SOURCE` and `GRAM_BCP_VAL_SOURCE` only with the intended existing
train and test files. It never substitutes test data for training. The full source
question sets are checked for overlap before deterministic seed-42 sampling;
source hashes and sampled indices are saved.

The smoke selects four train questions and two test questions, performs two
full-parameter updates with batch 2 x rollout 2, uses a 12K working context,
and validates/saves at step 2. Each episode has at most 16 actions and 8192
actor output tokens. These reduced budgets test the RL path, not BC-P quality.
The actor uses external corpus search/open, graph-memory actions and answer
submission through the same BC-P GRAM executor as zero-shot evaluation.
Training reward is **binary BC-P task reward + 0.1 x mean format reward**;
actor KL remains `low_var_kl`, coefficient `0.001`. This is a BC-P adaptation
of GRAM's document-QA objective, not a reproduced paper training setting.

The BC-P judge reuses existing `.openai_env` credentials. Reference answers
are private grading inputs, absent from actor/helper tasks. Retrieval, helper
and judge errors abort the run rather than silently becoming incorrect answers.
`task_reward` reports answer correctness separately from shaped training reward.

Logs live under `$SCRATCH/context-graph-gram/runs/gram-rl-smoke-JOBID-XXXXXX/`;
watch `suite.log`, `search.log`, `memory-server.log`, `ray-{2,3}.log`, and
`trainer.log`. The audit requires finite optimizer/KL metrics at both steps,
a nonzero gradient, both ranks' model/optimizer/extra-state shards, helper
activity, and BC-P search/grading traces in both training and validation.
Only then is `GRAM_RL_SMOKE_COMPLETE` written. Local tests cover configuration,
data isolation, reward routing, failure propagation and launch topology; a live
Vista run is still needed to verify GPU memory fit and successful updates.

### Four-node document-stream RL mechanics smoke

Both four-node RL smoke modes explicitly set `VLLM_USE_FLASHINFER_SAMPLER=0`
before starting each trainer's Ray daemon and in the PPO Ray runtime environment.
This uses vLLM's native sampler to bypass the observed FlashInfer sampling JIT
failure (`curand.h` absent from the compiler's include search path during model
profiling). It does not repair the CUDA installation or disable other FlashInfer
kernels. No conda packages are installed or replaced. Sampling backend changes
can change seeded token draws; the smoke is an execution check, not a controlled
performance comparison to earlier runs.

After data/dependency preparation, each trainer runs a small GPU top-k/top-p
sampling probe before any helper, retriever or Ray service starts. It verifies
that the installed vLLM honors the switch and selects `forward_native`, then
checks sampled token IDs and synchronizes CUDA. The launch stops on probe failure
or a 180-second timeout. `sampler-INDEX.log` and `sampler-HOST.json` record the
backend, package versions and GPU; expect `GRAM_NATIVE_SAMPLER_OK` on every
trainer. This does not certify full model initialization or optimizer updates.

`scripts/smoke_gram_rl_qwen35_9b_4node.sbatch` provisions a separate frozen
memory helper on node 0 and a private three-GPU Ray training cluster on nodes
1--3. Submit it with `sbatch`, or execute it with `bash` inside an **idle**
four-node idev allocation. It never stops an existing Ray cluster or allocation;
occupied GPUs/Ray nodes are rejected. It reuses the current Qwen3.5 training
environment (`deepseek_v4`, overridable with `TRAIN_CONDA_ENV`) without installing
packages or changing `cxtgraph`.

Because other jobs may be using the main checkout, launch from a detached
worktree. From a Vista login node:

```bash
cd /work/09281/chc_1996/vista/context-graph && git fetch origin && GRAM_CODE=$(mktemp -d "$SCRATCH/gram-rl-code-XXXXXX") && git worktree add --detach "$GRAM_CODE" origin/master && mkdir -p "$GRAM_CODE/logs" && PROJECT_ROOT="$GRAM_CODE" sbatch --chdir="$GRAM_CODE" "$GRAM_CODE/scripts/smoke_gram_rl_qwen35_9b_4node.sbatch"
```

For an existing idle allocation, replace the final `sbatch --chdir=...` invocation
with `PROJECT_ROOT="$GRAM_CODE" bash "$GRAM_CODE/scripts/smoke_gram_rl_qwen35_9b_4node.sbatch"`.
The batch request is four GH nodes for two hours; idev keeps its existing limit.
Do not run it within either occupied SWE allocation.

The smoke performs **two full-parameter optimizer updates**, with 12K context,
one question per batch, three sampled episodes, actor KL `0.001`, and a maximum
of 16 actions/8192 actor output tokens per episode. These reduced settings are
for mechanics, not the paper's training protocol. All ten context documents are
retained. The two bundled HotpotQA **validation** questions are assigned separate
smoke roles: row 0 for optimization, row 1 for validation. Their source split,
IDs and hashes are recorded; this is not a proper training dataset or benchmark
performance measurement. No dataset download or external judge is needed.

Artifacts are in `$SCRATCH/context-graph-gram/runs/gram-rl-smoke-JOBID-XXXXXX/`:
`preparation.log`, `memory-server.log`, `ray-{1,2,3}.log`, `trainer.log`,
`traces/`, `checkpoints/`, and `smoke-audit.json`. The completion marker requires
step-1/2 finite optimizer and KL metrics, at least one nonzero gradient norm,
nonempty model/optimizer/extra-state shards for all three ranks at step 2, and
actual actor/helper trace events. Zero-gradient runs fail this gate rather than
claim useful learning. Checkpoint reload, GPU memory fit and live Vista success
are not established by local tests; the audit explicitly does not certify reload
or performance. Watch `suite.log` and `trainer.log` for the actual result.

### General training entry

Prepare distinct train and validation Parquet files with the command above.
The preflight checks hashes, declared splits and task/question overlap. Serve a
**separate frozen memory checkpoint** first. Activate the established `cxtgraph`
training environment; this script neither modifies it nor cancels any jobs.

```bash
MODEL_PATH=/path/to/actor GRAM_TRAIN_DATA=/path/to/train/data.parquet GRAM_VAL_DATA=/path/to/validation/data.parquet GRAM_MEMORY_ENDPOINT=http://FROZEN_NODE:18000 GRAM_MEMORY_MODEL=FROZEN_MODEL_NAME GRAM_MEMORY_REVISION=CHECKPOINT_SHA GRAM_FROZEN_MEMORY=true bash scripts/train_gram.sh
```

For semantic merging during training append Hydra overrides
`+actor_rollout_ref.rollout.plugin.gram.embedding_endpoint=http://EMBED_NODE:8000`
and `+actor_rollout_ref.rollout.plugin.gram.embedding_model=NAME`; preserve that
service's checkpoint provenance with the run. Set `GRAM_NNODES`,
`GRAM_GPUS_PER_NODE`, and `GRAM_TP` to an **already provisioned** cluster. This is
not an all-in-one Vista sbatch launcher; GPU memory fit and optimizer updates on
Vista remain to be tested. Set `GRAM_TRACE_DIR` to a shared output path for workers.

Each sampled actor step becomes an exact-prefix VERL segment. All segments from
one episode share `gen_uid` and the same terminal reward. The existing
`foldgrpo / relative_extrema` path deduplicates these segments before calculating
group mean/std. Its token process masks are zero, so this is terminal GRPO with
no FoldAgent shaping and no clipping of the reward's possible `1.1` value. Wrong
or unfinished episodes remain in the group. The existing token-level loss
aggregation is retained; sequence weighting is not independently reproduced.

## Verification boundary

### BC-P XML decoding repair

The BC-P zero-shot and RL smoke launchers now default to
`GRAM_ACTION_DECODING=xml_regex`. The observed unconstrained failure was ordinary
prose preceding a valid action: the strict parser rejected 99 of 100 actions,
leaving the episode at `max_steps` with an empty answer and an unconsumed search
observation. Disabling tokenizer thinking alone does not constrain ordinary prose.

The new mode requests vLLM regex-constrained generation: optional tagged reasoning
and one XML action. Per-state action choices mirror executor guards: no external
retrieval while a document is pending, no insert/update without a document, and
no BC-P answer before an external search. It does not choose evidence or force a
final answer. The strict XML parser, raw responses, sampled tokens and returned
log-probabilities remain intact; no post-hoc tag extraction or answer fabrication
is performed. Token-budget truncation and poor action selection can still fail.
Both modern structured-output and legacy guided-decoding adapters forward regexes;
an unsupported server produces a request error rather than a silent fallback.

This changes the decoding protocol and can raise format reward mechanically; it
is not evidence that RL learned formatting or that the original paper was
reproduced. `config.action_decoding` is saved with evaluation manifests/episodes,
the grammar source is hashed, and RL overrides/traces record the same setting.
The shared executor and direct evaluator retain `unconstrained` as their default;
use `GRAM_ACTION_DECODING=unconstrained GRAM_BCP_PROGRESS_LIMIT=0` in launchers or
`--action-decoding unconstrained` in the evaluator to opt out. Use a fresh run
directory; incompatible manifests cannot resume into historical results.

First run the zero-shot launcher with `GRAM_SAMPLES=2`. Inspect
`evaluation/summary.json` for `answered`, `blank_predictions`,
`documents_consumed`, and `mean_format_reward_graded`, and inspect
`trajectory.jsonl` for actual `memory_call` events. These describe execution,
not answer correctness; do not launch all 150 rows based on HTTP 200s alone.
CPU tests cover the request wiring, state constraints, strict parsing, mocked
end-to-end search/helper/answer flow and preservation of training tokens/logprobs.
Actual Vista vLLM grammar compilation, model behavior and GPU training remain
live smoke checks.

### BC-P memory-search loop guard

The two-row XML-decoding rerun exposed a separate policy failure: `bcp-28`
performed one corpus search, consumed one document, then spent 98 actions on
internal `memory_search`. `bcp-6` made 73 internal searches and eventually
answered immediately after requesting fresh external evidence. Format compliance
alone did not establish useful agent progress.

BC-P smoke launchers now default to `GRAM_BCP_PROGRESS_LIMIT=2`. This opts into
an additional adaptation rule, shared by evaluation and RL: disable internal
search on an empty graph; allow at most two internal searches between successful
external retrieval or document consumption; process a pending document before
answering; and perform the first corpus search before opening a page. Merely
changing the query, emitting malformed XML, or attempting a blocked action does
not reset the counter. The decoder's allowed-action list and executor both
enforce it. Prompts record the last internal query and returned path count so
the actor can distinguish missing stored evidence from external search results.

Set `GRAM_BCP_PROGRESS_LIMIT=0` to reproduce the previous XML-only behavior.
The direct executor/evaluator retain the disabled default; direct evaluation
opts in with `--action-decoding xml_regex --bcp-progress-limit 2`. The limit is
saved in manifests/episodes and RL overrides; blocked actions stay in raw traces
and training segments, with `valid_format=true`, `executed=false` and an error.
Format reward still measures syntax, not semantic progress. No extra generation
budget or fabricated final answer is added. External-search loops, poor evidence,
wrong answers, and token-truncated XML remain possible. The guard is a protocol
change, not a paper-faithful reproduction or evidence of improved BC-P accuracy.
Rerun the same two seeded rows in a fresh output directory before scaling up.

### BC-P external request history and empty edits

The `d35ef4d` two-row run reached 100 steps on both rows with an empty final
graph: 97 consumed observations, 97 empty edits, and 98 external searches.
One row issued the same query 50 times. This demonstrates an external request
loop; it does not establish why the helper extracted no facts. Inspect the
saved `memory_call` requests and responses to distinguish empty entities from
empty relations and unsupported actor proposals.

With `bcp_progress_limit > 0`, the actor now sees the last eight external
requests (request text capped at 512 characters, up to five returned docids)
and the last edit's extraction status and actual graph edge changes. This is
operational metadata, not an alternative answer-evidence store. Empty edits
explicitly report that no new facts were saved. Re-inserting an existing triple
does not count as a new edge even when it adds a new source attribution.

The executor rejects a previously executed external request before calling the
retriever. Search keys normalize case and whitespace; open-page keys retain
case-sensitive docids. Detection covers the whole episode, including requests
older than the bounded prompt history. Blocking does not advance the stream,
reset internal-search counters, or remove the sampled action from training.
The actor receives feedback to refine its query or open another source. This
extends the opted-in adaptation guard and changes the protocol; set the limit
to zero for previous unrestricted external retrieval. Eval source hashes prevent
resuming older results with the changed executor.

Episodes and summaries record `no_change_memory_edits` and
`duplicate_retrievals_blocked`; action audits include `memory_effect` and helper
edits distinguish `explicit_skip`, `no_entities`, and `no_relations`. Empty
extraction remains valid and does not fabricate facts or trigger an automatic
answer. This fix prevents duplicate tool execution and restores missing actor
feedback; it does not guarantee useful extraction, completion, or accuracy.

### Frozen helper JSON schema

An RL run passed GPU sampling preflight and reached its first rollout, then
failed because the helper returned a relation row that was not three strings.
The helper now requests vLLM `structured_outputs.json` for all three operations:
entities are an array of nonempty strings, relations are an array of exactly
three-string arrays, and maintenance is an object with only required `add` and
`remove` arrays of triples. Empty arrays remain valid when there is no evidence.
The schema does not add facts. Local shape validation,
entity membership checks, removal validation and atomic graph updates remain
authoritative. No malformed rows are silently dropped or repaired. Truncation,
invalid content or an unsupported schema server still fails visibly.

Every helper audit records its requested schema, original response, finish reason
and usage. Evaluation manifests record `gram-memory-json-schema-v2-entity-enum`; RL traces
record the schema per call. This changes helper decoding, so use a new run and
retain the prior artifacts. Both evaluation and RL use the shared helper.

RL launchers run a three-request synthetic helper format probe after helper
startup but before loading the training models. It emits `GRAM_MEMORY_SCHEMA_OK`
only after all operations return valid shapes and the synthetic entities and
relation preserve the explicit Alice/Paris fact. Valid but empty entity JSON
now fails this positive control. Responses are saved separately
in `memory-schema-preflight.jsonl` and are excluded from training trace counts;
the synthetic probe is not benchmark evidence. Failures abort launch and retain
the response audit. Local tests cover schemas, request forwarding, unchanged
graphs on failures and probe failure propagation. Live grammar enforcement and
successful RL updates remain Vista checks.

The `38f1513` RL rerun exposed another schema gap: three-string triples could
still contain endpoints absent from the extracted entity list. Relation requests
now use positional `prefixItems` with an enum of that request's extracted names
for subject and object; the middle relation remains a nonempty string. Empty
relation lists remain valid. Local membership validation still fails atomically
if a server violates this constraint, and errors include the unexpected names.
The synthetic preflight and saved-request probe exercise this schema before any
training restart. Maintenance can add new source-supported entities and therefore
retains its separate shape schema; existing graph/removal checks still apply.
See the [JSON Schema array specification](https://json-schema.org/understanding-json-schema/reference/array)
for positional tuple constraints. Actual Vista backend compilation remains a
live preflight check, not a result established by CPU schema validation.

### Empty entity extraction diagnosis

The saved `IivxVy` run contains 97 helper calls, all `entities`, all returning
`[]` with two completion tokens and `finish_reason=stop`. The supplied source
snippets contain named entities and supported intermediate clues; no relation
request was made. This localizes the failure to entity generation before graph
construction, rather than token truncation or dropping triples. It does not
alone distinguish empty-list copying, over-filtering for final answerability,
or following the retrieval tool's appended instructions.

The revised extraction prompt explicitly distinguishes the naming reference
`existing_entities` from the output, preserves intermediate clues without
requiring an answer to the full question, and separates source-supported facts
from fallible actor proposals. Unsupported proposals must not suppress valid
facts in the same document. Relation extraction also distinguishes document
metadata dates from dates of life events. Empty outputs remain allowed; schemas
do not force invented entities or relations.

`scripts/probe_gram_helper_idev.sh` uses one idle allocated GPU for a frozen
helper, replaying the first saved entities request per attempt from
`GRAM_HELPER_SOURCE`. It starts no actor, corpus server, judge, or trainer.
`scripts/probe_gram_helper.py` compares the exact saved messages with three
single-field ablations (existing names, question, exact tool footer), revised
prompt alone, original prompt plus schema, and revised prompt plus schema.
It then exercises the production entity-to-relation graph edit and synthetic
positive/negative controls. The positive source has a useful fact despite an
unanswerable question and an unsupported actor proposal; the negative source
has no evidence for the actor's proposed fact.

`helper-probe.jsonl` preserves requests and raw responses; `helper-probe.json`
and `probe.log` show the comparisons. Exit 0 requires nonempty graph edits on
the saved cases and passing synthetic controls; this is not a benchmark score
or a grounding audit of every extracted triple. The old run is read-only, and
the probe refuses an occupied helper endpoint or an existing output audit.
The live `helper-probe-1047279-qhLVNF` run then reproduced `[]` for both original
requests. Removing only `question` produced 44 and 67 entities; removing only
`existing_entities` or the tool footer did not. Applying a schema to the original
prompt also left both outputs empty. This isolates the independent question field
as a cause of empty entity extraction for these two saved requests under the tested
model/decoding setup. It does not establish the same mechanism for every document
or for relation extraction.

The longer revised prompt was insufficient: the first case extracted 24 entities
but produced no relations, the second hit the completion budget, and the synthetic
positive control produced an empty graph. These results are failed extraction
checks, not evidence of a working memory graph or successful evaluation.

The production edit payload now omits the independent research `question` for
entity, relation and maintenance calls. The actor still sees the question and
selects `requested_facts`; the helper receives those proposals and the unmodified
source document, validating facts rather than solving the research question.
The relation instruction now selects from requested facts without asking it to
filter by question. No minimum number of entities/triples is forced, and no
truncated output is accepted. Manifests identify this input contract as
`gram-memory-source-facts-v1` separately from the JSON output protocol.

The updated probe retains all seven earlier variants and adds the current prompt
without the question, both with and without schema. It also tests relations with
and without the question while holding the prompt, document and extracted entity
enum fixed. `production_calls` and `control_calls` expose raw stage responses and
finish reasons in the report, including failures. The positive control verifies a
birthplace relation, not merely any edge between Alice and Paris.

RL preflight now runs the same production edit pipeline and positive/negative
controls instead of supplying a prefilled entity list to independent schema calls.
`memory-semantic-preflight.json` records both results even on failure; failed
controls prevent training startup. Maintenance still has a separate format probe.
CPU tests establish request isolation and failure gates. The new payload's live
extraction quality, truncation behavior and end-to-end eval/RL remain unverified
until the updated one-GPU probe passes and subsequent jobs complete.

### Offline search cache on a full `/work` volume

The `foAhRI` evaluation stopped before helper requests: `load_dataset` found the
prepared corpus but failed creating its builder lock with `OSError: [Errno 28]`.
This is a cache-write failure, not evidence about the revised helper's behavior.

The shared Vista `_search` launcher (used by GRAM evaluation and BC-P RL) now sets
`SEARCH_CORPUS_CACHE_READ_ONLY=1`. It reads the latest prepared default corpus
snapshot under `HF_DATASETS_CACHE` (normally `$SEARCH_HF_HOME/datasets`) using
[`Dataset.from_file`](https://huggingface.co/docs/datasets/loading#arrow), without
constructing a dataset builder. Model and embedding Hub cache locations stay the
same. It neither copies the corpus nor deletes caches. `SEARCH_CORPUS_CACHE_DIR`
can pin a specific directory containing `dataset_info.json` and its Arrow shards;
`SEARCH_CORPUS_CACHE_READ_ONLY=0` restores the regular builder path.

Snapshot identity, required columns, shard lengths and total train rows are
checked before use. `SEARCH_CORPUS_CACHE_READ_ONLY` in `search.log` records the
resolved snapshot and ordered shards. Missing/incomplete caches fail explicitly;
there is no download or partial-corpus fallback. The server retains its existing
docid, URL and snippet conversion, including no extra title prepending for this
HF corpus path. Local parquet mode still takes precedence when explicitly set.

CPU tests compare real single/sharded Arrow fixtures against the normal HF cache
builder, simulate `ENOSPC` on lock acquisition, and verify identical corpus indexes
and unchanged source cache files. They do not establish Vista service startup,
helper quality or successful RL updates.

### Existing verification

`python -m pytest tests/test_gram.py` exercises document advancement, invalid
actions, graph transactions/provenance, multi-hop retrieval, cosine merging,
helper errors, reference isolation, dataset export, exact sampled training
prefixes, and episode-deduplicated GRPO. CPU tests establish these mechanics;
live model behavior, distributed training and paper-score reproduction need
separate runs with saved manifests and trajectories.
The Windows CPU test environment does not contain the full training stack
(`peft`/`cachetools` were missing during trainer/worker import probes); only
configuration composition and the actor-to-training-output adapter were verified
locally, not a Ray worker startup or optimizer update.
