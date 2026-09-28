# Qwen3.5-9B GAIA internal comparison

This uses the prepared text-only GAIA validation parquet and the existing local
BrowseComp-Plus search corpus. It is not an official GAIA evaluation: there is
no live web or attachment tool support. No training is performed. Report the
actual row count and data hash from the manifest rather than assuming 127 rows.

Use one existing four-node Vista idev allocation per method, after earlier
evaluations in that allocation have finished. The launcher shares the BC-P
allocation lock, serving fixes, 32K budget, seed 42 and scratch checkpoint.
It defaults to all rows; `SAMPLES=8` selects a smoke subset.

Update the shared checkout once before starting either method:

```bash
git pull --ff-only origin master
```

In the ContextGraph allocation:

```bash
bash scripts/eval_gaia_qwen35_9b_4node_idev.sh contextgraph
```

In the FoldAgent allocation:

```bash
bash scripts/eval_gaia_qwen35_9b_4node_idev.sh foldagent
```

Default data: `data/gaia_validation.parquet`. Both methods use that same file;
the evaluator overrides the workflow per method. `DATA_PATH` can select another
prepared GAIA parquet; non-GAIA rows, attachments, missing labels and duplicate
task IDs fail preflight before GPU services start. Do not silently substitute
the RL training or holdout split for the validation set.

Results and service logs stream to `outputs/gaia-local-qwen35-9b-{method}-{job}-{timestamp}`.
Each manifest records the benchmark, retrieval protocol, data path/hash, exact
checkpoint and code revision. `GAIA_EVAL_COMPLETE` indicates successful merging
of all shards with no execution or judge parse failures. Allocation expiry can
still interrupt a run; check current `squeue` time remaining before launching.

The shared launcher preflights the tokenizer, judge, ordinary completion and
controller JSON requests. Local unit/syntax checks do not verify Vista GPU
execution; inspect the actual suite log for launch and completion status.
