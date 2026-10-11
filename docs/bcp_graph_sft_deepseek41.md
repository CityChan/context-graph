# BC-P explicit graph-policy SFT with DeepSeek V4.1

This collector runs the current `explicit_tools` research agent on **BC-P
training questions**. DeepSeek chooses ordinary research and graph tools; the
framework executes them against the BC-P corpus. There is no automatic graph
maintenance, graph-call quota, or synthetic successful tool feedback.

The default teacher is the hosted `deepseek-flash` API, documented as
DeepSeek-V4.1-Flash on 2026-10-11. The old V4-Flash-0731 launchers are not used.
The collector also accepts an already-running V4.1 vLLM endpoint. It does not
provision V4.1 weights or assume that the old V4 TP=4 configuration fits V4.1.

Sources: [model naming](https://api-docs.deepseek.com/api/responses),
[JSON mode](https://api-docs.deepseek.com/guides/json_mode/),
[thinking control](https://api-docs.deepseek.com/guides/thinking_mode/).

## Vista smoke

Use an allocated compute node with one free GPU for corpus retrieval, or set
`LOCAL_SEARCH_URL` to an existing BC-P retrieval service. The wrapper activates
`cxtgraph` and only cleans up the retrieval process it starts. The teacher is
remote; this path does not need a four-node model allocation.

Set `DEEPSEEK_API_KEY` in the environment. `OPENAI_API_KEY`, `OPENAI_BASE_URL`
and `JUDGE_MODEL` remain dedicated to the existing BC-P answer grader. Without
a real judge key, scoring is explicitly **strict EM only**; this can reject
correct aliases. Teacher credentials are never copied into the grader.

```bash
cd /work/09281/chc_1996/vista/context-graph
git pull --ff-only
SAMPLES=3 CANDIDATES=2 bash scripts/generate_bcp_graph_sft_deepseek41.sh
```

Default inputs are `data/bc_train.parquet` and `data/bc_test.parquet`. The latter
is only used to reject overlapping questions, never for generation. Selection
is seeded and deduplicated before `START_INDEX`/`SAMPLES` slicing. Validation
is split from the training source by question hash, keeping all candidates,
branches and compaction segments of a question together across shards.

`DRY_RUN=1` validates datasets/configuration without retrieval or model calls.
`PREFLIGHT_ONLY=1` checks retrieval health and two teacher JSON action responses.
`STUDENT_TOKENIZER_PATH` can point to a cached Qwen3.5-9B tokenizer. Defaults:
32K total context, 24K cumulative response/observation allowance, 8K branch
allowance, 100 model turns, 2,048 output tokens per ordinary request.

For an existing vLLM teacher, set `TEACHER_TRANSPORT=vllm`,
`DEEPSEEK_BASE_URL` (including `/v1`) and `DEEPSEEK_MODEL` to its exact served
model ID. Set `DEEPSEEK_API_KEY` if that endpoint requires authentication.
No automatic downgrade to another model is performed. API model aliases are
not immutable checkpoints: requested and returned model identifiers are logged.

## Collection and export

Each question has two independently sampled full episodes by default. These
are **not** matched counterfactual continuations from saved mid-task states.
The existing conditional graph-use prompt is identical to explicit evaluation.
References stay in the environment/grader; the teacher receives the question
and retrieved evidence. Non-thinking mode collects JSON actions for the same
non-thinking student interface, not hidden reasoning labels.

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
