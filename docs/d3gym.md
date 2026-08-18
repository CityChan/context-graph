# D3-Gym integration

This integration runs the existing ReAct, FoldAgent, and ContextGraph code
loops against the official per-task D3-Gym images. Agent Python executes at
`/task`; task inputs are discovered below `/task/datasets`; generated artifacts
persist under `/task/pred_results`; and reward comes from the image's
`eval_script.eval()` result.

## Prepare metadata

From the login node with Hugging Face access:

```bash
cd /work/09281/chc_1996/vista/context-graph
conda run -n cxtgraph python scripts/make_d3gym_data.py --out-dir data --runtime auto
```

The builder creates a deterministic repository-disjoint train/validation split
and emits `code`, `code_branch`, and `code_graph` parquets.

## Cache task images

The official D3-Gym Docker images currently use the `linux/amd64`
architecture. They do not run natively on Vista GH200 (`arm64`) compute nodes.
The launcher inspects each SIF before loading the model and exits immediately
on an architecture mismatch. Run official images on x86_64 GPU compute nodes,
or use native arm64 images if the benchmark authors publish them. QEMU
emulation is not suitable for multi-turn training because each tool call starts
a task container and is many times slower than native execution.

Load Apptainer if the executable is not already on `PATH`, then cache the
images needed by the chosen split. For a one-task smoke test:

```bash
python scripts/cache_d3gym_images.py --parquet data/d3gym_val_code.parquet --image-dir $SCRATCH/d3gym_images --runtime apptainer --limit 1
```

For training, pass both train and validation parquets and omit `--limit`.
Images are named `$D3GYM_IMAGE_DIR/task_N.sif`; the launcher checks them before
starting Ray so missing images do not waste a GPU allocation.

## Run

Inside an existing four-node allocation with architecture-compatible images:

```bash
bash scripts/smoke_d3gym_qwen3_30b_instruct_4node.sh
```

For scheduled eight-node jobs:

```bash
D3GYM_MODE=smoke D3GYM_METHOD=react sbatch scripts/run_d3gym_30b_instruct_8node.sh
D3GYM_MODE=train D3GYM_METHOD=fold sbatch scripts/run_d3gym_30b_instruct_8node.sh
D3GYM_MODE=train D3GYM_METHOD=ctxgraph sbatch scripts/run_d3gym_30b_instruct_8node.sh
```

To submit the three 30B training variants together:

```bash
bash scripts/submit_d3gym_30b_train_suite.sh
```

Use `D3GYM_RUNTIME=docker` on a host with Docker. The `local` backend accepts
unpacked task directories and is intended for development/tests.
