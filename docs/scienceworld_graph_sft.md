# ScienceWorld explicit graph SFT with local DeepSeek

Use a separate existing four-node GH200 idev allocation. This entry point reuses
the local DeepSeek Ray/vLLM launcher, with TP=4 across the four GPUs. It starts
no BC-P retrieval service and needs no DeepSeek cloud API key. The existing
`MODEL_PATH` and `MODEL_ID` overrides are preserved; the default is the cached
DeepSeek-V4-Flash-0731 checkpoint, not a claim that those weights are V4.1.

```bash
cd /work/09281/chc_1996/vista/context-graph
git pull --ff-only
AGENT_BUDGET_MODE=working_context SAMPLES=3 CANDIDATES=2 bash scripts/generate_scienceworld_graph_sft_deepseek_4node_idev.sh
```

This launches six candidate episodes sequentially (`WORKERS=1`). Defaults are
32K working context, 100 model turns, 100 environment steps, and 2,048 output
tokens per request. In `working_context` mode, rendered input plus output must
fit the window; lifetime token consumption is recorded but is not capped at
24K. Graph compaction can reclaim window space. Explicit graph calls consume
the same turn allowance as task actions. Use a fresh run directory after changing
the budget mode or code; old rejected candidates are not resumed under new settings. The teacher
runs in `deepseek_v4`; the collector uses `cxtgraph` plus the existing isolated
ScienceWorld 1.2.3 dependency overlay and Java runtime.

ScienceWorld uses `focus_v3`, unsimplified official **train** variations, and
`explicit_tools`. Automatic graph pruning/consolidation/retrieval memory is
disabled. The model chooses graph operations and environment commands. The
simulator ends episodes; `finish` is not available to the model. Candidate
generation does not force graph calls or manufacture successful feedback.

The endpoint wrapper now defaults to `SFT_SELECTION=task_balanced`: shuffle
within task types and take a seeded round-robin across types before applying
`START_INDEX`/`SAMPLES`. Small batches therefore cover different task types
when available, rather than repeatedly sampling only conductivity variations.
This uses training metadata, not task outcomes or gold solutions. It changes
the collection distribution; it is not a representative accuracy evaluation.

`SFT_TEACHER_PROFILE=graph_demonstration_v1` adds collection-only guidance to
the initial system prompt: preserve useful evidence and failed attempts, consider
merge/prune after meaningful phases, and request compaction before running out
of space. The model still selects every action and target, and may decline graph
work. There is no forced graph turn, automatic pruning, or operation quota.
The exact guidance is retained in sampled prefixes and exported SFT messages;
formal evaluation does not enable it by default. Record the prompt difference
when interpreting teacher/student results. Accepted Parquets record the profile
and selection mode. `TEACHER_TEMPERATURE` defaults to 0.7.

To reproduce the former collection strategy, explicitly set
`SFT_TEACHER_PROFILE=evaluation SFT_SELECTION=uniform`. Use a fresh run directory
for any strategy change. These generation changes do not guarantee successful
tasks or useful graph trajectories; inspect diagnostics and accepted-row counts
before scaling up. Success and graph-evidence review gates are unchanged.

Dataset metadata is generated under the run directory, without overwriting
shared repository datasets. Before generation, every training task/variation
is checked against the simulator's official training split. Identity and
train/validation grouping use task name plus variation index, not task wording.
The official dev metadata is an overlap guard; no dev/test episodes are run.
Export validation rows are held out by identity from the collected training
source, with all candidates and branches of an identity in the same split.

The common collector and curator are `generate_bcp_graph_sft.py --benchmark
scienceworld` and `build_bcp_graph_sft.py`; their historical filenames do not
mean this path runs BC-P. Output is the same multi-turn JSON-action Parquet
format as [BC-P](bcp_graph_sft_deepseek41.md). In addition to clean protocol and
graph trace checks, ScienceWorld requires native task reward 1, `completed=1`
and `environment_natural_end=1`. After graph work, there must be a task action
or branch return. Merge/prune require a separate teacher evidence review.
Successful no-graph controls are kept separately. Zero graph training rows
causes a nonzero exit with rejection counts, not silent gate relaxation.

Default output is printed at startup, under
`$SCRATCH/contextgraph_sft/deepseek_v4_flash_0731_interactive/<job-tag>/scienceworld/explicit`:

- `manifest.json`: checkpoint metadata hashes, simulator version, budgets, commit.
- `episodes/*/episode.json`: completed candidate records, including failures to solve.
- `episodes/*/attempt-*/requests.jsonl`, `trajectory.json`, `tools.jsonl`: raw evidence.
- `sft-*/`: accepted Parquets, `decisions.jsonl`, curation manifest and review requests.

If every candidate is rejected, diagnose saved artifacts before another GPU run:

```bash
python scripts/diagnose_graph_sft.py "$RUN_DIR"
```

This needs no teacher or simulator. It reports termination reasons and budget
statistics, tools present in request schemas versus tools actually sampled,
repeated commands, recent environment feedback and excerpts of model-visible
messages. New runs also save `diagnostics.json` before curation, including when
no SFT rows survive. `status=ok` means the candidate was recorded, not solved.
The raw `requests.jsonl` remains the source for full, unabridged contexts.

Set `RUN_DIR` to resume completed candidates with identical manifest settings and
commit. No accepted row count is guaranteed. Trace auditing and teacher review
do not prove causal benefit from graph operations. Local tests use a fake
simulator/teacher with the real graph executor; a Vista smoke is still needed
to verify actual checkpoint serving and native ScienceWorld execution together.
