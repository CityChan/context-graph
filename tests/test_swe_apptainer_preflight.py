"""Exercise batch artifact/failure handling without pretending to run a container."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
BATCH = ROOT / "scripts/preflight_swe_apptainer_vista.sbatch"
PROBE = ROOT / "scripts/check_swe_apptainer_sympy.sh"
BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(not BASH, reason="Bash is required")


def run_batch(tmp_path, failure="", job=True):
    scratch = tmp_path / "scratch"
    scratch.mkdir(exist_ok=True)
    setup = tmp_path / "mock-tools.sh"
    setup.write_text("""module() { :; }
uname() { echo aarch64; }
flock() { :; }
apptainer() {
    case "$1" in
        --version) echo 'apptainer mock';;
        pull) [[ "$PROBE_FAILURE" != pull ]] || return 31; printf 'mock-sif' > "$2";;
        exec) [[ "$PROBE_FAILURE" != tests ]] || return 32; echo 'mock container success';;
        *) return 99;;
    esac
}
""", encoding="utf-8", newline="\n")
    env = dict(os.environ, BASH_ENV=setup.as_posix(), PROJECT_ROOT=ROOT.as_posix(),
               SCRATCH=scratch.as_posix(), PROBE_FAILURE=failure)
    env.pop("SLURM_JOB_ID", None)
    if job:
        env["SLURM_JOB_ID"] = "12345"
    return subprocess.run([BASH, BATCH.as_posix()], cwd=ROOT, env=env,
                          text=True, capture_output=True, timeout=30,
                          creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)


def test_batch_publishes_pinned_image_and_separate_run_artifacts(tmp_path):
    for _ in range(2):
        result = run_batch(tmp_path)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "SWE_APPTAINER_PREFLIGHT_COMPLETE" in result.stdout
    root = tmp_path / "scratch/context-graph-swe"
    assert len(list((root / "images").glob("*.sif"))) == 1
    assert len(list((root / "runs").glob("*/preflight-complete.txt"))) == 2
    for record in (root / "runs").glob("*/provenance.txt"):
        assert "@sha256:a8b2a526" in record.read_text()
        assert "base_commit=cffd4e0f86fefd4802349a9f9b19ed70934ea354" in record.read_text()
    for record in (root / "runs").glob("*/preflight-complete.txt"):
        assert "evaluation_performed=false" in record.read_text()


@pytest.mark.parametrize("failure,code,stage", [("pull", 31, "image_pull"), ("tests", 32, "repository_and_tests")])
def test_batch_fails_without_publishing_completion(tmp_path, failure, code, stage):
    result = run_batch(tmp_path, failure)
    assert result.returncode == code, result.stdout + result.stderr
    assert f"FAILED stage={stage}" in result.stdout
    assert not list(tmp_path.glob("scratch/context-graph-swe/runs/*/preflight-complete.txt"))


def test_corrupt_cached_image_is_rejected(tmp_path):
    assert run_batch(tmp_path).returncode == 0
    image = next(tmp_path.glob("scratch/context-graph-swe/images/*.sif"))
    image.write_text("corrupt")
    result = run_batch(tmp_path)
    assert result.returncode != 0
    assert "FAILED stage=image_pull" in result.stdout
    assert len(list(tmp_path.glob("scratch/context-graph-swe/runs/*/preflight-complete.txt"))) == 1


def test_batch_requires_allocation_and_shell_syntax_is_valid(tmp_path):
    result = run_batch(tmp_path, job=False)
    assert result.returncode != 0
    assert "Submit this script with sbatch" in result.stderr
    for path in (BATCH, PROBE):
        result = subprocess.run([BASH, "-n", path.as_posix()], capture_output=True,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        assert result.returncode == 0, result.stderr
