# Qwen3.5-9B BC-P evaluation

The 9B wrapper shares the Qwen3.8 launcher and its Vista CUDA, compiler and
node-local cache fixes. It selects `Qwen/Qwen3.5-9B` for both server and client,
resolves its own scratch snapshot (ignoring an inherited `MODEL_PATH`), and
writes `outputs/bcp-qwen35-9b-{method}-{job}-{timestamp}`. Manifests record the
model ID and exact snapshot. No dependencies are installed or upgraded.

From the repository on a Vista login node, submit the download:

```bash
sbatch scripts/download_qwen35_9b_vista.sbatch
```

Wait for `CHECKPOINT_READY` in `download-qwen35-9b-JOBID.log`. Weights live in
`$SCRATCH/hf_cache/hub`, not `/work`. Then use separate existing four-node idev
allocations, one per method:

```bash
bash scripts/eval_bcp_qwen35_9b_4node_idev.sh foldagent
```

```bash
bash scripts/eval_bcp_qwen35_9b_4node_idev.sh contextgraph
```

Defaults match the 27B protocol: 8 questions, seed 42, repaired memory, 32K
context (8K prompt plus 24K response budget), thinking enabled, two workers per
replica, one retriever and three TP=1 model replicas. Both methods must run the
same commit and snapshot. Do not update shared code during an active run.
After both smoke runs finish with zero execution/judge parse errors, set
`SAMPLES=-1` before the same commands for all 150 rows.

The existing allocation lock is shared across models to prevent port/resource
collisions. See [the common launcher documentation](bcp_qwen38_eval.md) for
credentials, preflights and artifact details. Local tests do not establish
successful GPU execution or model accuracy; those require the Vista smoke run.
