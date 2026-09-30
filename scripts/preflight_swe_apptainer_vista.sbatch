#!/bin/bash
#SBATCH --job-name=swe-arm-preflight
#SBATCH --partition=gg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --account=AST24021
#SBATCH --chdir=/work/09281/chc_1996/vista/context-graph
#SBATCH --output=logs/swe-arm-preflight.%j.out
#SBATCH --error=logs/swe-arm-preflight.%j.err

# Infrastructure probe only: no agent, gold patches, grading, or model server.
set -euo pipefail
run_label=${SLURM_JOB_ID:-direct}
: "${SCRATCH:?SCRATCH must be set}"
PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
cd "$PROJECT_ROOT"
scratch_root=$(realpath -e "$SCRATCH")
root="$scratch_root/context-graph-swe"
# Refuse pre-existing directory symlinks that would redirect large artifacts.
for path in "$root" "$root/images" "$root/runs" "$root/apptainer-cache"; do
    case "$(realpath -m "$path")/" in "$scratch_root/"*) ;; *) echo "Path escapes SCRATCH: $path" >&2; exit 1;; esac
done
mkdir -p "$root/images" "$root/runs" "$root/apptainer-cache" logs
run_dir=$(mktemp -d "$root/runs/preflight-${run_label}-XXXXXX")
exec > >(tee -a "$run_dir/suite.log") 2>&1
stage=environment
trap 'rc=$?; if (( rc != 0 )); then echo "SWE_APPTAINER_PREFLIGHT_FAILED stage=$stage exit=$rc artifacts=$run_dir"; fi' EXIT
echo "SWE Apptainer infrastructure preflight run=$run_label node=$(hostname) artifacts=$run_dir"
export GOMAXPROCS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
[[ $(uname -m) == aarch64 ]] || { echo "This probe requires an aarch64 node"; exit 1; }
module load tacc-apptainer/1.4.1
apptainer --version
unset APPTAINER_BIND APPTAINER_BINDPATH SINGULARITY_BIND SINGULARITY_BINDPATH
export APPTAINER_CACHEDIR="$root/apptainer-cache"
export APPTAINER_TMPDIR="$run_dir/tmp"
mkdir -p "$APPTAINER_TMPDIR" "$run_dir/worktree"

# Confirmed in the pinned Verified dataset c104f840cc67f8b6eec6f759ebc8b2693d585d4a.
# Epoch supplies this ARM image on a best-effort basis; this job tests it on Vista.
instance=sympy__sympy-20590
base_commit=cffd4e0f86fefd4802349a9f9b19ed70934ea354
digest=a8b2a5265717391b168a5d7aa884b466e80fa748d074373a56d328cc48d887b9
image_uri="docker://ghcr.io/epoch-research/swe-bench.eval.arm64.$instance@sha256:$digest"
image_name="$instance-arm64-$digest.sif"
image="$root/images/$image_name"
stage=image_pull
echo "Image=$image_uri"
(
    flock -w 600 9
    cd "$root/images"
    if [[ -e "$image_name" || -e "$image_name.sha256" ]]; then
        sha256sum --check "$image_name.sha256"
    else
        # Build from the pinned OCI image so squashfs resource limits are explicit.
        apptainer build --mksquashfs-args "-processors 1 -mem 256M" "$run_dir/image.sif" "$image_uri"
        mv -- "$run_dir/image.sif" "$image"
        sha256sum "$image_name" > "$image_name.sha256"
    fi
) 9> "$root/images/$image_name.lock"
cp -- "$root/images/$image_name.sha256" "$run_dir/image.sha256"
printf 'instance=%s\nbase_commit=%s\nimage_uri=%s\nimage_path=%s\n' "$instance" "$base_commit" "$image_uri" "$image" > "$run_dir/provenance.txt"

stage=repository_and_tests
# This runs only our fixed probe, not generated/untrusted agent commands.
# Disable site-wide host binds; expose only the fresh worktree and read-only probe.
apptainer exec --cleanenv --containall --no-mount hostfs,bind-paths --bind "$run_dir/worktree:/workspace" --bind "$PROJECT_ROOT/scripts/check_swe_apptainer_sympy.sh:/probe.sh:ro" --pwd /workspace "$image" /bin/bash --noprofile --norc /probe.sh "$base_commit"
stage=complete
printf 'instance=%s\npublic_baseline_tests=passed\nevaluation_performed=false\n' "$instance" > "$run_dir/preflight-complete.txt"
echo "SWE_APPTAINER_PREFLIGHT_COMPLETE instance=$instance artifacts=$run_dir evaluation_performed=false"
