#!/bin/bash
# Fixed public baseline probe, executed inside the pinned ARM image.
set -euo pipefail
base_commit=${1:?Expected base commit}
[[ "$base_commit" =~ ^[0-9a-f]{40}$ ]] || exit 2
[[ $(uname -m) == aarch64 ]] || { echo "Wrong container architecture"; exit 1; }
[[ -d /testbed/.git ]] || { echo "Image has no /testbed Git repository"; exit 1; }
[[ -z $(find /workspace -mindepth 1 -maxdepth 1 -print -quit) ]] || { echo "Probe requires an empty worktree"; exit 1; }
# SIF is immutable. Copy into the job-owned writable bind, with current ownership.
cp -a --no-preserve=ownership /testbed/. /workspace/
cd /workspace
git checkout --detach "$base_commit"
git reset --hard "$base_commit"
git clean -ffd
[[ $(git rev-parse HEAD) == "$base_commit" ]]
[[ -z $(git status --porcelain) ]]
# Remove future refs/objects before any later use of this disposable worktree.
git for-each-ref --format='delete %(refname)' | git update-ref --stdin
git reflog expire --expire=now --all
git gc --prune=now
set +u
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate testbed
set -u
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python --version
python -c 'import pathlib, sympy; p=pathlib.Path(sympy.__file__).resolve(); assert pathlib.Path("/workspace") in p.parents, p; print("SymPy import:", p, "version:", sympy.__version__)'
# Existing public tests only; not the hidden task tests or a resolved-rate score.
timeout 600 python bin/test sympy/core/tests/test_basic.py
echo "SWE_APPTAINER_PUBLIC_TESTS_PASSED"
