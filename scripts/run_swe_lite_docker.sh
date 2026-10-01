#!/bin/bash
# End-to-end SWE-bench Lite evaluation on a Linux x86_64 Docker host.
set -euo pipefail
method=${1:?Expected react, foldagent, or contextgraph}
case "$method" in react|foldagent|contextgraph) ;; *) echo "Invalid method: $method" >&2; exit 2;; esac
case "$(uname -sm)" in "Linux x86_64"|"Linux amd64") ;; *) echo "Official SWE-bench images require a Linux x86_64 Docker host" >&2; exit 2;; esac
cd "$(dirname "$0")/.."
agent_python=${SWE_AGENT_PYTHON:-.venv-swe-agent/bin/python}
grade_python=${SWE_GRADE_PYTHON:-.venv-swe-grade/bin/python}
model_path=${SWE_MODEL_PATH:-cache/qwen35-9b-tokenizer}
endpoint=${SWE_ENDPOINT:-http://127.0.0.1:18000}
data_dir=${SWE_LITE_DATA_DIR:-data/swebench-lite}
samples=${SWE_SAMPLES:-5}
workers=${SWE_WORKERS:-1}
context_length=${SWE_CONTEXT_LENGTH:-65536}
for path in "$agent_python" "$grade_python"; do [[ -x "$path" ]] || { echo "Missing Python environment: $path" >&2; exit 2; }; done
[[ -d "$model_path" ]] || { echo "Missing matching local tokenizer: $model_path" >&2; exit 2; }
case "$samples" in -1|[1-9]|[1-9][0-9]*) ;; *) echo "SWE_SAMPLES must be -1 or positive" >&2; exit 2;; esac
case "$workers" in [1-9]|[1-9][0-9]*) ;; *) echo "SWE_WORKERS must be positive" >&2; exit 2;; esac
docker info --format '{{.OSType}} {{.Architecture}}' | grep -Eq '^linux (x86_64|amd64)$' || { echo "Linux x86_64 Docker daemon required" >&2; exit 2; }
"$agent_python" -c 'import json,sys,urllib.request; model=json.load(urllib.request.urlopen(sys.argv[1].rstrip("/")+"/v1/models",timeout=15))["data"][0]; need=int(sys.argv[2]); have=int(model["max_model_len"]); assert have>=need, f"vLLM max_model_len={have} < requested={need}"; print(f"SWE_MODEL_CONTEXT_OK max_model_len={have} requested={need}")' "$endpoint" "$context_length"
if [[ ! -e "$data_dir" ]]; then
  "$agent_python" scripts/eval_swebench_verified.py prepare --dataset lite --data-dir "$data_dir"
fi
run="outputs/swe-lite-${method}-$(date +%Y%m%d_%H%M%S)-$$"
mkdir -p outputs logs
exec > >(tee -a "logs/$(basename "$run").log") 2>&1
echo "SWE_LITE_RUN_START method=$method samples=$samples context=$context_length output=$run"
"$agent_python" -u scripts/eval_swebench_verified.py generate --dataset lite --method "$method" --data-dir "$data_dir" --model-path "$model_path" --endpoint "$endpoint" --context-length "$context_length" --samples "$samples" --workers "$workers" --output "$run"
"$grade_python" -u scripts/eval_swebench_verified.py grade --dataset lite --data-dir "$data_dir" --output "$run" --workers "$workers"
echo "SWE_LITE_RUN_COMPLETE output=$run summary=$run/grading/summary.json"
