#!/bin/bash
# One real model-generated patch and ARM grading, in an existing compute allocation.
set -euo pipefail
: "${SLURM_JOB_ID:?Run inside idev or a compute job}"
: "${SCRATCH:?SCRATCH must be set}"
case "$(hostname -s)" in login*) echo 'Run this on the compute node'; exit 2;; esac
cd "${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}"
method=${1:-contextgraph}
case "$method" in contextgraph|foldagent|react) ;; *) echo 'Expected contextgraph, foldagent, or react'; exit 2;; esac
root="$(realpath -e "$SCRATCH")/context-graph-swe"
agent_env=${SWE_AGENT_ENV:?Set SWE_AGENT_ENV to the dedicated agent venv printed by setup}
agent_python="$agent_env/bin/python"
endpoint=${SWE_ENDPOINT:-http://127.0.0.1:18000}
curl --connect-timeout 5 --max-time 15 -fsS "$endpoint/v1/models"
set +u
module load tacc-apptainer/1.4.1
set -u
export PYTHONNOUSERSITE=1 PYTHONPATH="$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$root/runs"
run=$(mktemp -d "$root/runs/arm-pilot-${SLURM_JOB_ID}-XXXXXX")
exec > >(tee -a "$run/suite.log") 2>&1
echo "SWE ARM pilot: method=$method endpoint=$endpoint artifacts=$run"
stage=grading_environment
trap 'rc=$?; if (( rc != 0 )); then echo "SWE_ARM_PILOT_FAILED stage=$stage exit=$rc artifacts=$run"; fi' EXIT
# Dedicated grading environment, without inherited GPU runtimes.
grade_env="$agent_env/grading"
if [[ ! -x "$grade_env/bin/python" ]]; then "$agent_python" -m venv "$grade_env"; fi
"$grade_env/bin/python" -m pip install -r requirements_swebench_eval.txt
data="$root/data/verified-c104f840"
stage=calibration
"$grade_env/bin/python" scripts/grade_swe_arm_pilot.py --data-dir "$data" --apptainer-root "$root" --output "$run/calibration" --calibrate
stage=generation
model="$SCRATCH/hf_cache/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
# Apply the same Vista Torch TLS preload used by the working model server.
preload=$("$agent_python" -c 'import importlib.util,pathlib; print(pathlib.Path(importlib.util.find_spec("torch").origin).parent / "lib/libtorch_global_deps.so")')
[[ -s "$preload" ]] || { echo "Missing Torch TLS preload: $preload"; exit 2; }
LD_PRELOAD="$preload${LD_PRELOAD:+:$LD_PRELOAD}" "$agent_python" scripts/eval_swebench_verified.py generate --backend apptainer --apptainer-root "$root" --data-dir "$data" --output "$run/generation" --method "$method" --model-path "$model" --endpoint "$endpoint" --instance-ids sympy__sympy-20590 --samples 1 --workers 1
stage=grading
"$grade_env/bin/python" scripts/grade_swe_arm_pilot.py --data-dir "$data" --apptainer-root "$root" --output "$run/generation"
stage=complete
echo "SWE_ARM_RUN_COMPLETE artifacts=$run (one task; ARM compatibility result)"
