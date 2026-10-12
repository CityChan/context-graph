# BC-P explicit graph-policy SFT with DeepSeek V4.1

This collector runs the current `explicit_tools` research agent on **BC-P
training questions**. DeepSeek chooses ordinary research and graph tools; the
framework executes them against the BC-P corpus. There is no automatic graph
maintenance, graph-call quota, or synthetic successful tool feedback.

The default shell transport is a **local vLLM teacher**, with no cloud API key.
The existing DeepSeek Slurm launcher can start the checkpoint and retrieval
service and then run this collector. It retains its `MODEL_PATH`, `MODEL_ID`,
node layout and serving environment. Its existing default checkpoint is
DeepSeek-V4-Flash-0731, not V4.1: a filename or served alias does not upgrade
weights. Use the actual V4.1 path/model configuration when that is the teacher.
Checkpoint path, configuration/index hashes and model architecture are recorded
in the collection manifest; weight files themselves are not hashed.

Sources: [model naming](https://api-docs.deepseek.com/api/responses),
[JSON mode](https://api-docs.deepseek.com/guides/json_mode/),
[thinking control](https://api-docs.deepseek.com/guides/thinking_mode/).

## Vista smoke

With the existing local checkpoint configured in the DeepSeek sbatch launcher:

```bash
cd /work/09281/chc_1996/vista/context-graph
git pull --ff-only
BCP_EXPLICIT_GRAPH_SFT=1 SAMPLES=3 CANDIDATES=2 sbatch scripts/generate_ctxgraph_sft_deepseek_v4_flash_0731_9node.sh
```

This launcher defaults to nine nodes (eight for the teacher, one for retrieval).
Keep the working `MODEL_PATH`/`MODEL_ID` and topology overrides from your existing
submission. Do not assume the older model's node count or vLLM options fit a new
checkpoint. The server uses `deepseek_v4`; collection uses `cxtgraph`. Local
collection output defaults to `$ARTIFACT_ROOT/bcp-explicit`. `PREFLIGHT_ONLY=1`
on this Slurm launcher checks local serving dependencies without starting services.
The normal run also probes teacher action responses before collecting episodes.

`OPENAI_API_KEY`, `OPENAI_BASE_URL`
and `JUDGE_MODEL` remain dedicated to the existing BC-P answer grader. Without
a real judge key, scoring is explicitly **strict EM only**; this can reject
correct aliases. Teacher credentials are never copied into the grader.

For already-running services, set `DEEPSEEK_BASE_URL`, `DEEPSEEK_MODEL` and
`LOCAL_SEARCH_URL`, then run `scripts/generate_bcp_graph_sft_deepseek41.sh` with
Bash. Without an existing search URL, this endpoint wrapper needs a free GPU
and starts its own retrieval process. Hosted API use is optional and requires
explicit `TEACHER_TRANSPORT=deepseek` and `DEEPSEEK_API_KEY`.

Default inputs are `data/bc_train.parquet` and `data/bc_test.parquet`. The latter
is only used to reject overlapping questions, never for generation. Selection
is seeded and deduplicated before `START_INDEX`/`SAMPLES` slicing. Validation
is split from the training source by question hash, keeping all candidates,
branches and compaction segments of a question together across shards.

On the endpoint wrapper, `DRY_RUN=1` validates datasets/configuration without retrieval or model calls.
`PREFLIGHT_ONLY=1` checks retrieval health and two teacher JSON action responses.
`STUDENT_TOKENIZER_PATH` can point to a cached Qwen3.5-9B tokenizer. Defaults:
32K working context, 100 model turns, 2,048 output tokens per ordinary request.
The default `AGENT_BUDGET_MODE=working_context` limits each rendered input plus
output to the physical window, including branches. Lifetime consumption is
recorded, not capped at 24K; compaction can reclaim window space. Use a fresh
run directory after changing the budget mode or code, rather than resuming
old rejected candidates under different settings.

For an existing vLLM teacher, set `TEACHER_TRANSPORT=vllm`,
`DEEPSEEK_BASE_URL` (including `/v1`) and `DEEPSEEK_MODEL` to its exact served
model ID. Set `DEEPSEEK_API_KEY` if that endpoint requires authentication.
No automatic downgrade to another model is performed. API model aliases are
not immutable checkpoints: requested and returned model identifiers are logged.

## Collection and export

Each question has two independently sampled full episodes by default. These
are **not** matched counterfactual continuations from saved mid-task states.
The endpoint wrapper defaults to `SFT_TEACHER_PROFILE=graph_demonstration_v1`,
which adds collection-only guidance for evidence notes, responding to failed
actions and model-requested compaction before window exhaustion. Actions,
arguments, retained evidence and operation timing remain model choices; no
graph-call quota or automatic graph operation is introduced. Formal evaluation
keeps its existing prompt. Use `SFT_TEACHER_PROFILE=evaluation` to reproduce the
former evaluation-prompt collection strategy. `TEACHER_TEMPERATURE` defaults to
0.7; these settings are stored in the manifest. Use a fresh run directory when
changing them. The shared Python CLI defaults to the evaluation profile unless
`--teacher-profile` is supplied, as the wrappers do.
References stay in the environment/grader; the teacher receives the question
and retrieved evidence. Non-thinking mode collects JSON actions for the same
non-thinking student interface, not hidden reasoning labels.

The demonstration guidance remains in the exact sampled conditioning prefixes
and exported SFT messages; it is not silently stripped. Accepted rows record
`teacher_profile` and `selection`. Validate the student with the intended eval
prompt after SFT; guided teacher demonstrations are not zero-shot eval results,
and guidance does not guarantee accepted graph trajectories.

Every request saves its exact messages, raw response, permitted action schema,
teacher usage, requested/returned model IDs and finish reason. API output is
re-tokenized with the student tokenizer for episode accounting, not represented
as the teacher's original token IDs or log probabilities. Hosted JSON mode
constrains JSON syntax only; the local action contract checks arguments before
dispatch. vLLM mode additionally requests schema-constrained decoding. A
successful preflight is not a universal format guarantee.

Per-episode results are written atomically under `episodes/`; failed and
interrupted attempts remain in separate attempt directories. Set `RUN_DIR` to
the same path to resume with exactly matching manifest settings and commit.
Already completed episodes (including wrong answers) are skipped; failed
attempts can be retried. A new `sft-*` directory preserves prior exports.

The curator requires:

- Correct answer, natural finish, no protocol/graph/grading errors or budget stop.
- Hash-continuous graph trace with expected operation effects; graph calls must
  match sampled model responses, not environment bookkeeping.
- A task action/return/finish following graph work.
- A separate conservative teacher review of every merge/prune against the
  recorded pre-edit evidence. Oversized review inputs are rejected, not truncated.
- Exact sampled conditioning prefixes for every assistant training label.
  Branches and compaction are separate chats; long segments are dropped whole.

This produces `graph_train.parquet`, `control_train.parquet` and corresponding
`*_validation.parquet` files **only when those buckets are nonempty**. The
standard multi-turn SFT loader consumes `messages`, `tools`, `enable_thinking`;
use `truncation=error` and a matching maximum length. System/user evidence is
not a training target. All segments of a graph episode, including its later
task continuation, stay in the graph dataset. Controls are successful episodes
with no graph-maintenance tool calls (they may branch), saved separately so their
mixture is an explicit training decision. Review requests, episode rejection reasons and
counts are saved alongside the Parquets. Zero surviving graph training rows
causes a nonzero exit after the report is written; it is not a successful data
generation run. A small smoke may have no validation rows.

Evidence review uses the same teacher in a separate request, not an independent
human judgment. Trace validation checks saved transitions, not a fresh replay
of retrieval or proof of factual preservation. These are **SFT candidates**:
sample the merge/prune reviews manually before a large training run. Neither
answer success nor the review establishes causal graph benefit. Matched-budget
ablations and student validation are still needed.

For re-curation without rerunning the environment:

```bash
python scripts/build_bcp_graph_sft.py "$RUN_DIR" --output "$RUN_DIR/sft-reviewed" --tokenizer Qwen/Qwen3.5-9B --review
```

Use a fresh output directory. `manifest.json` records row counts and the input
manifest hash; `decisions.jsonl` identifies rejected episodes. Do not lower the
gates merely because the teacher continues to avoid graph tools.
