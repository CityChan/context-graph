# Route A self-distillation

This is a separate SFT route based on the supplied Route A specification. The
`huancheng/graphrpo-compress-r1` source branch was not available on this checkout's
origin when implemented; this is not a cherry-pick or a claim of byte-identical
reproduction. Existing DeepSeek auto-teacher collection and curation are unchanged.

## Collection

Inside an **idle four-node Vista idev allocation**, run:

```bash
git pull --ff-only
SAMPLES=3 CANDIDATES=2 bash scripts/generate_route_a_sft_qwen35_9b_4node_idev.sh scienceworld
```

Use `bcp` instead of `scienceworld` for BC-P. Use `SAMPLES=-1` for all eligible
training tasks. The default is **65,536 tokens of rendered input plus output per
request**, not a lifetime episode cap. It uses the cached Qwen3.5-9B checkpoint
as both policy and tokenizer, with thinking enabled, the evaluation auto/controller
configuration, and actual simulator/retrieval actions. No DeepSeek teacher key is
needed. BC-P retains its separate judge credentials. Official train variations
and held-out exclusion are checked before launching GPU services.

The launcher uses one policy GPU, a separate BC-P retrieval GPU when needed, and
CPU collection on a third node. It does not claim four-GPU inference throughput.
The existing server environment (`deepseek_v4`) supplies vLLM; collection uses
`cxtgraph` and the **Qwen** checkpoint. Neither environment is replaced. Override
`QWEN35_9B_MODEL_PATH` to select an explicit cached snapshot.

`ROUTE_A_RUN_DIR` overrides the stable collection root. Resume only with the same
commit, dataset, checkpoint, tokenizer and generation parameters. Completed
candidates, including failed-to-solve episodes, are not silently resampled.
Candidate seeds differ; sampling defaults to temperature 0.7. Controller sampling
retains the evaluation configuration. A lock prevents duplicate collectors.

## Acceptance and supervision

An episode must succeed and finish naturally, with a valid graph trace, matched
sampled decisions and no recorded protocol/graph errors, timeout or budget stop.
ScienceWorld additionally requires native successful completion. This route does
not require a model evidence reviewer or proof of causal benefit from graph edits.

- `graph`: successful episodes containing a model-selected graph edit (not pass).
- `control`: successful episodes without such an edit, including controller pass.
- Heuristic prune/bind actions remain trace evidence and are never invented as
  assistant targets.

The `/v1/completions` transport sends the agent's actual input IDs and requires
vLLM's native output IDs. Thinking, special tokens and EOS are preserved in the
labels. API logprobs are not collected; these artifacts are **offline SFT data**,
not reusable on-policy RL rollouts.

Prefix-compatible requests from the same context are merged into segments only
when both the token prefix and message prefix agree. A history rewrite starts a
new segment. Historical messages have `train=false`; newly sampled assistant
responses have `train=true`. Each response is supervised once, under its original
prefix. `agent_context` stores native IDs, loss mask, sampled spans and prefix/output
hashes. A tokenizer vocabulary/special-token/template fingerprint is checked at
training time. The message text is audit metadata; it is not re-rendered to create
different conditioning tokens. Overlength requests are rejected, never truncated.

Exports contain:

- `graph_train.parquet`, `control_train.parquet` and corresponding validation files
  when nonempty. Group names refer to the source episode; graph rows can include
  ordinary actor actions as well as controller decisions.
- `train.parquet`: all graph training rows plus at most
  `floor(CONTROL_RATIO * graph_rows)` control rows, default `CONTROL_RATIO=0.5`.
- `validation.parquet`: all accepted validation rows, with no training mixture cap.
- `manifest.json` and `decisions.jsonl`: accepted/rejected episodes, exact counts,
  source identity, file hashes, `training_ready` and `validation_ready`.

Task-hash splitting groups all candidates and segments of a task together. A tiny
smoke can export validation only or control only; that does not make it training
ready. Full collection requires completion and nonempty graph training rows.

## Include assisted explicit + thinking data

The original `RouteA_plus_thinking` recipe also used previously curated assisted
explicit + thinking demonstrations. Keep these as a separately identified source:
Route A supplies auto/controller rollouts; assisted data supplies explicit-tool
demonstrations with its original assistance and reasoning context. The mixed
dataset does not relabel explicit examples as controller rollouts.

Use `scripts/mix_route_a_sft.py` with the **original source tokenizer**. Both
sources must have the same vocabulary, special tokens and chat template. Existing
source quality/held-out curation is a prerequisite: registration records the
operator's source designation and file hashes, it does not verify task success
or rediscover whether an arbitrary dataset was framework-assisted.

Supported assisted formats:

- `messages_train`: saved messages with explicit boolean `train` on **every**
  message, `enable_thinking=true`, and canonical `question_hash` (or `task_id`
  containing that same hash). Each marked assistant answer becomes a separate
  supervised request with its saved prefix. All earlier answers have zero loss.
  Thinking text/reasoning fields must survive the template. These rows are labeled
  `legacy_messages_retokenized`; original sampled-token equality cannot be claimed
  without the original request records.
- `agent_context`: the version-1 token/mask/span format described above, including
  the matching tokenizer fingerprint. Existing IDs and masks are validated and
  retained. Other historical `agent_context` formats need a source-specific
  adapter; the import does not guess their semantics.

Example in a Python environment with pandas, pyarrow and the checkpoint's
transformers support (replace the paths with the actual exports):

```bash
python scripts/mix_route_a_sft.py --register-assisted --assisted-manifest /scratch/assisted-source.json --assisted-train /scratch/old/train.parquet --assisted-validation /scratch/old/validation.parquet --tokenizer /scratch/qwen35-checkpoint --assisted-format messages_train
python scripts/mix_route_a_sft.py --route-a /scratch/route-a/sft-export --assisted-manifest /scratch/assisted-source.json --tokenizer /scratch/qwen35-checkpoint --output /scratch/route-a-plus-thinking --assisted-ratio 1.0
```

Preserve the old train/validation assignment; conflicting task hashes across
either source's train and validation sets cause an error, even if downsampling
would have hidden the conflict. Duplicate supervised prefix/output pairs are
removed within a split and rejected across splits. Overlength examples are
rejected without truncation.

The default `--assisted-ratio 1.0` caps assisted **rows** at one per Route A
training row, without repeating data. This is a configurable starting mix, not
a recovered historical recipe or a token-level 50/50 split. Route A's exported
graph/control mixture is preserved. The manifest reports actual row counts and
supervised token counts per source, source hashes, and removed duplicates.
All accepted validation rows are retained independently of this sampling cap.

Point `ROUTE_A_SFT_DIR` at the **mixed output directory** to use the existing
training command below. No change to the training loss-mask mode is needed.
Actual historical data and its source metadata are still needed to verify the
import; fixture tests alone do not establish compatibility with that export.

## Training

On an idle four-node allocation, set `ROUTE_A_SFT_DIR` to the printed export and
`QWEN35_9B_MODEL_PATH` to the original collection checkpoint, then run:

```bash
bash scripts/train_route_a_sft_qwen35_9b_4node_idev.sh
```

This defaults to **one full-parameter optimizer step**, not a completed training
experiment. It checks all training rows with `loss_mask_mode=agent_context` and
`truncation=error`, then uses the existing FSDP2 trainer. Both training and
validation exports must exist. Training uses `deepseek_v4` by default; formal
evaluation remains in `cxtgraph`. `TOTAL_TRAINING_STEPS` and other explicit training
overrides select longer runs. A live 64K GPU memory/optimizer smoke is still required;
local tests do not establish that the requested training batch fits Vista GPUs.

This implements offline SFT only. The controller's timing remains framework-owned.
The existing `not is_train` guard for online-RL historical replacement is retained:
enabling it safely also requires sampled policy logprobs and graph-credit/turn
mapping to stay aligned after context rewrites. SFT correctness does not establish
that separate RL contract or autonomous graph-operation timing.
