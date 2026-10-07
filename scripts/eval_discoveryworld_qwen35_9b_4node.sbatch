#!/bin/bash
#SBATCH --job-name=discoveryworld-9b
#SBATCH --partition=gh
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=48:00:00
#SBATCH --account=AST24021
#SBATCH --chdir=/work/09281/chc_1996/vista/context-graph
#SBATCH --output=logs/discoveryworld-9b.%j.out
#SBATCH --error=logs/discoveryworld-9b.%j.err
set -euo pipefail
: "${SCRATCH:?SCRATCH must be set}"
export PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
export BENCH_DIFFICULTY=${BENCH_DIFFICULTY:-Normal}
case "$BENCH_DIFFICULTY" in Easy|Normal|Challenge|all) ;; *) echo 'Invalid BENCH_DIFFICULTY'; exit 2 ;; esac
export BENCH_PROMPT_PROFILE=discoveryworld_v1
export BENCH_REQUIREMENTS=requirements_discoveryworld.txt
BENCH_BASE_PYTHON=$(python "$PROJECT_ROOT/scripts/select_discoveryworld_python.py")
export BENCH_BASE_PYTHON
python_tag=$("$BENCH_BASE_PYTHON" -c 'import sys; print("py%d%d" % sys.version_info[:2])')
export BENCH_AGENT_ENV=${BENCH_AGENT_ENV:-$SCRATCH/context-graph-agent-benchmarks/envs/discoveryworld-fd591323-$python_tag}
echo "DISCOVERYWORLD_AGENT_PYTHON=$BENCH_BASE_PYTHON overlay=$BENCH_AGENT_ENV"
export BENCH_DATA=${BENCH_DATA:-$SCRATCH/context-graph-agent-benchmarks/data/discoveryworld-fd591323-$BENCH_DIFFICULTY/tasks.json}
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1
exec bash "$PROJECT_ROOT/scripts/eval_agent_benchmarks_qwen35_9b_4node.sbatch" discoveryworld "${1:-both}"
