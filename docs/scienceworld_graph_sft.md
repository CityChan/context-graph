# ScienceWorld explicit graph SFT with local DeepSeek

Use a separate existing four-node GH200 idev allocation. This entry point reuses
the local DeepSeek Ray/vLLM launcher, with TP=4 across the four GPUs. It starts
no BC-P retrieval service and needs no DeepSeek cloud API key. The existing
`MODEL_PATH` and `MODEL_ID` overrides are preserved; the default is the cached
DeepSeek-V4-Flash-0731 checkpoint, not a claim that those weights are V4.1.

```bash
cd /work/09281/chc_1996/vista/context-graph
git pull --ff-only
SAMPLES=3 CANDIDATES=2 bash scripts/generate_scienceworld_graph_sft_deepseek_4node_idev.sh
```

This launches six candidate episodes sequentially (`WORKERS=1`). Defaults are
32K working context, 24K cumulative response/observation budget, 100 model
turns, 100 environment steps, and 2,048 output tokens per request. Explicit
graph calls consume the same turn/token allowance as task actions. The teacher
runs in `deepseek_v4`; the collector uses `cxtgraph` plus the existing isolated
ScienceWorld 1.2.3 dependency overlay and Java runtime.

ScienceWorld uses `focus_v3`, unsimplified official **train** variations, and
`explicit_tools`. Automatic graph pruning/consolidation/retrieval memory is
disabled. The model chooses graph operations and environment commands. The
simulator ends episodes; `finish` is not available to the model. Candidate
generation does not force graph calls or manufacture successful feedback.

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

Set `RUN_DIR` to resume completed candidates with identical manifest settings and
commit. No accepted row count is guaranteed. Trace auditing and teacher review
do not prove causal benefit from graph operations. Local tests use a fake
simulator/teacher with the real graph executor; a Vista smoke is still needed
to verify actual checkpoint serving and native ScienceWorld execution together.
