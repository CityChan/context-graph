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
ready. Full collection requires completion and nonempty graph training rows. This
does not mix in older thinking datasets automatically: their tokenizer, sampled
prefixes and held-out separation must be independently established first.

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
