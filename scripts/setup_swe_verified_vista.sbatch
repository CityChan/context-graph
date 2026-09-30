#!/bin/bash
#SBATCH --job-name=swe-env-setup
#SBATCH --partition=gg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --account=AST24021
#SBATCH --chdir=/work/09281/chc_1996/vista/context-graph
#SBATCH --output=logs/swe-env-setup.%j.out
#SBATCH --error=logs/swe-env-setup.%j.err

# Prepare infrastructure only. The Apptainer agent/grader adapter is still pending.
set -euo pipefail
run_label=${SLURM_JOB_ID:-direct}
: "${SCRATCH:?SCRATCH must be set}"
export PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
cd "$PROJECT_ROOT"
[[ $(uname -m) == aarch64 ]] || { echo "Requires Vista aarch64"; exit 2; }
scratch_root=$(realpath -e "$SCRATCH")
root="$scratch_root/context-graph-swe"
for path in "$root" "$root/envs" "$root/data" "$root/runs"; do
    case "$(realpath -m "$path")/" in "$scratch_root/"*) ;; *) echo "Path escapes SCRATCH: $path"; exit 2;; esac
done
mkdir -p "$root/envs" "$root/data" "$root/runs"
run_dir=$(mktemp -d "$root/runs/setup-${run_label}-XXXXXX")
exec > >(tee -a "$run_dir/suite.log") 2>&1
stage=python_environment
trap 'rc=$?; if (( rc != 0 )); then echo "SWE_VISTA_SETUP_FAILED stage=$stage exit=$rc artifacts=$run_dir"; fi' EXIT
git rev-parse HEAD > "$run_dir/commit.txt"
echo "Preparing SWE infrastructure on $(hostname); artifacts=$run_dir"
# Reuse the installed ARM PyTorch through system-site-packages. pip writes only
# into a dedicated venv, never into the RL environment.
base_python=${SWE_BASE_PYTHON:-/work/09281/chc_1996/vista/miniconda3/envs/deepseek_v4/bin/python}
if [[ -n "${SWE_AGENT_ENV:-}" ]]; then
    venv_dir=$(realpath -e "$SWE_AGENT_ENV")
    case "$venv_dir/" in "$root/envs/"*) ;; *) echo "SWE_AGENT_ENV must be inside $root/envs"; exit 2;; esac
    [[ -f "$venv_dir/pyvenv.cfg" && -x "$venv_dir/bin/python" ]] || { echo "SWE_AGENT_ENV must be an existing venv"; exit 2; }
else
    venv_dir=$(mktemp -d "$root/envs/agent-${run_label}-XXXXXX")
    "$base_python" -m venv --system-site-packages "$venv_dir"
fi
echo "SWE agent environment: $venv_dir"
printf 'python=%s\n' "$venv_dir/bin/python" > "$run_dir/paths.txt"
export PYTHONNOUSERSITE=1 PYTHONPATH="$PROJECT_ROOT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE
export HF_HOME="$root/hf-cache" PIP_CACHE_DIR="$root/pip-cache"
"$venv_dir/bin/python" -m pip install --only-binary=:all: -r requirements_swebench_generation.txt
"$venv_dir/bin/python" -m pip freeze > "$run_dir/python-freeze.txt"
stage=preparation_environment
# Do not inherit GPU libraries. Transformers 5 can import installed Torch even
# for AutoTokenizer, and no model runtime is needed to prepare this dataset.
prepare_env="$venv_dir/preparation"
if [[ ! -d "$prepare_env" ]]; then
    "$base_python" -m venv "$prepare_env"
fi
"$prepare_env/bin/python" -m pip install --only-binary=:all: -r requirements_swebench_prepare.txt
"$prepare_env/bin/python" -m pip freeze > "$run_dir/preparation-freeze.txt"
echo "SWE preparation environment: $prepare_env"
stage=data_and_tokenizer
data_dir="$root/data/verified-c104f840"
model_path="$scratch_root/hf_cache/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
(
    flock -w 600 9
    "$prepare_env/bin/python" scripts/prepare_swe_verified_vista.py --data-dir "$data_dir" --model-path "$model_path" --report "$run_dir/environment.json"
) 9> "$root/data/verified.lock"
printf 'python=%s\npreparation_python=%s\ndata_dir=%s\nmodel_path=%s\n' "$venv_dir/bin/python" "$prepare_env/bin/python" "$data_dir" "$model_path" > "$run_dir/paths.txt"
image_preflight=pending
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    stage=arm_image_and_public_tests
    bash scripts/preflight_swe_apptainer_vista.sbatch
    image_preflight=passed
else
    echo "Direct setup: Python/data/tokenizer prepared; ARM image conversion and public tests remain pending."
fi
stage=complete
printf 'environment_prepared=true\nimage_preflight=%s\nevaluation_performed=false\napptainer_agent_backend_ready=false\n' "$image_preflight" > "$run_dir/setup-complete.txt"
if [[ "$image_preflight" == passed ]]; then
    echo "SWE_VISTA_SETUP_COMPLETE artifacts=$run_dir"
else
    echo "SWE_VISTA_PYTHON_DATA_READY image_preflight=pending artifacts=$run_dir"
fi
