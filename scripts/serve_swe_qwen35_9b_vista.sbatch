#!/bin/bash
#SBATCH --job-name=swe-9b-server
#SBATCH --partition=gh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=01:00:00
#SBATCH --account=AST24021
#SBATCH --chdir=/work/09281/chc_1996/vista/context-graph
#SBATCH --output=logs/swe-9b-server.%j.out
#SBATCH --error=logs/swe-9b-server.%j.err

# Inference only. Repository containers and official tests run on an x86 Docker host.
set -euo pipefail
: "${SLURM_JOB_ID:?Run inside idev or submit with sbatch}"
: "${SCRATCH:?SCRATCH must be set}"
# Vista GH nodes have one GPU. Lock before loading weights or profiling KV cache;
# two concurrent launches can both pass the free-memory check and then OOM.
# Keep the descriptor across both execs for the lifetime of the API server.
server_lock="/tmp/contextgraph-swe-server-${UID}.lock"
exec 9>"$server_lock"
if ! flock -n 9; then
  echo "SWE_SERVER_ALREADY_RUNNING node=$(hostname -s) lock=$server_lock; inspect the existing server log" >&2
  exit 2
fi
# Older launchers do not hold the lock. Refuse to compete with their live engines
# (or another GPU workload); never kill a process from this launcher.
gpu_processes=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
if [[ -n "${gpu_processes//[[:space:]]/}" ]]; then
  echo "SWE_SERVER_GPU_BUSY node=$(hostname -s) compute_pids=$gpu_processes; inspect nvidia-smi before retrying" >&2
  exit 2
fi
export PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
export CONDA_SH=${CONDA_SH:-/work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh}
export SERVER_CONDA_ENV=${SERVER_CONDA_ENV:-deepseek_v4}
export MODEL_ID=Qwen/Qwen3.5-9B
export MODEL_REVISION=${MODEL_REVISION:-c202236235762e1c871ad0ccb60c8ee5ba337b9a}
export MODEL_PORT=${MODEL_PORT:-18000} WORKERS=${WORKERS:-2} SEED=${SEED:-42}
export MODEL_MAX_LEN=${SWE_MODEL_MAX_LEN:-65536}
case "$MODEL_MAX_LEN" in 32768|65536) ;; *) echo "Unsupported SWE model context: $MODEL_MAX_LEN"; exit 2;; esac
echo "SWE server max_model_len=$MODEL_MAX_LEN"
cd "$PROJECT_ROOT"
set +u
source "$CONDA_SH"
conda activate "$SERVER_CONDA_ENV"
set -u
export HF_HOME="$SCRATCH/hf_cache" HF_HUB_CACHE="$SCRATCH/hf_cache/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MODEL_PATH=${MODEL_PATH:-}
MODEL_PATH=$(python -c 'import os,json; from pathlib import Path; from huggingface_hub import snapshot_download; scratch=Path(os.environ["SCRATCH"]).resolve(); p=Path(os.environ["MODEL_PATH"] or snapshot_download(os.environ["MODEL_ID"],revision=os.environ["MODEL_REVISION"],cache_dir=str(scratch/"hf_cache"/"hub"),local_files_only=True)).resolve(); assert p.is_relative_to(scratch), "Checkpoint must be on SCRATCH"; idx=json.loads((p/"model.safetensors.index.json").read_text()); shards=set(idx["weight_map"].values()); assert shards and all((p/s).is_file() and (p/s).stat().st_size>0 for s in shards), "Incomplete checkpoint"; print(p)')
export MODEL_PATH
echo "Model=$MODEL_ID checkpoint=$MODEL_PATH"
echo "SWE model server node=$(hostname -f) port=$MODEL_PORT job=$SLURM_JOB_ID"
echo "Reach this endpoint from the Docker host through an SSH tunnel; allocation time limit applies."
# Reuse the CUDA/compiler/TLS/node-local-cache setup already used by the 9B evaluations.
# Its BC-P/GAIA dataset selector is irrelevant to inference. Keep the caller's
# benchmark in the evaluator, but do not pass it into the reused server launcher.
exec env -u BENCHMARK bash scripts/eval_bcp_qwen38_4node_idev.sh _server
