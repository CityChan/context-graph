# Auto ContextGraph teacher collection

Use the existing evaluation `controller` protocol to generate demonstrations.
This restores scheduled model graph decisions, automatic pruning/connection,
branch archival and working-history replacement. It does not use the
`framework_assisted_v1` explicit-tools scheduler. The collector obtains its
configuration from the BC-P / ScienceWorld evaluation configuration builders;
the manifest records the resolved settings.

Inside an **idle four-node Vista idev allocation**, run:

```bash
cd /work/09281/chc_1996/vista/context-graph
git pull --ff-only
SAMPLES=3 CANDIDATES=2 bash scripts/smoke_auto_graph_sft_deepseek_4node_idev.sh scienceworld
```

Replace `scienceworld` with `bcp` for BC-P. Each run samples three training
tasks and two candidates per task. ScienceWorld balances task families and
uses official unsimplified train variations. BC-P excludes held-out questions.
The launcher serves the cached DeepSeek-V4-Flash-0731 checkpoint in `deepseek_v4`
and collects in `cxtgraph`; no cloud teacher key is needed. Do not run on the
allocation occupied by SWE-bench Lite. Existing checkpoint overrides remain
available. The smoke defaults to 32K working context, 100 task turns, and a
single sequential collector. It does not establish equivalence to a historical
57% score, which requires matching the model, split, context and other settings.

`SFT_GRAPH_PROTOCOL=controller` selects the runtime protocol. The older
`EXPLICIT_GRAPH_SFT` / `BCP_EXPLICIT_GRAPH_SFT` launcher switches select the
durable per-episode collector rather than the legacy batch exporter; they no
longer determine the protocol within this new wrapper. The manifest must show
`profile=contextgraph_auto_teacher_v1`, `graph_action_protocol=controller`,
`structured_graph_controller=true`, `consolidation_interval=5`, and
`auto_prune_max_active=12`, and `enable_retrieval_memory=true`. Automatic history
replacement is enabled, even though session summaries
(`enable_summary`) remain disabled in these evaluation configurations. The
teacher chat endpoint uses non-thinking generation, recorded in the manifest.

## Saved evidence and SFT

Each episode retains `requests.jsonl`, `trajectory.json` with graph trace,
environment logs, native reward, termination and configuration provenance.
Diagnostics count graph events by source: `model`, `heuristic`, `environment`.
Automatic pruning/connect operations remain trace evidence. They are not
fabricated as model outputs. Controller-selected merge/prune/add_edge operations
retain the exact sampled JSON, candidate map and before/after states.

The auto curator requires task success, natural completion, valid graph trace,
matched sampled graph responses, a structural model edit, post-edit task
continuation, and clean execution. Merge/prune labels additionally require the
existing evidence review. Failed/unfinished episodes remain diagnostic artifacts.
An empty accepted set causes a nonzero exit, rather than reporting SFT success.

Outputs under the printed `sft-*` directory:

- `actor_train.parquet` / `actor_validation.parquet`: actual task-policy requests.
- `controller_train.parquet` / `controller_validation.parquet`: actual graph
  controller requests, with `timing_source=framework`.
- `manifest.json`, `decisions.jsonl`, `review-requests.jsonl`: counts and rejection
  reasons. Validation splits group all candidates and turns of a task together.

Rows use `messages`, `tools`, `enable_thinking=false`, and
`train_last_response_only=true`. Use this repository's `MultiTurnSFTDataset`
with `truncation=error` and sufficient `max_length`; it masks all historical
assistant turns. This is necessary because auto archival rewrites past user
messages. Each row supervises only the new response under its exact sampled
context. Overlength rows are rejected, never silently truncated. The actor and
controller datasets can be combined for SFT, but these demonstrations teach
scheduled controller decisions, not autonomous graph-operation timing.

Local mocked-environment tests validate collection/curation and loss masking.
Actual accepted counts and success rates require the live teacher smoke.
