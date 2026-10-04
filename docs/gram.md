# GRAM document-stream baseline

This is an independent, paper-guided implementation of **GRAM: Empowering Agent
with Actively Managed Graph-Structured Memory via Reinforcement Learning** (the
anonymous 16-page PDF supplied by the user). It does not change ContextGraph,
FoldAgent, their saved results, or the ScienceWorld protocol. GRAM's task here is
document-stream question answering, not a replacement ScienceWorld environment.
Source PDF SHA-256: `fc4843b1d5fa8680752e3944d4554e7eec3f0826be83cbdff1cd80e8b9846c3a`.

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
