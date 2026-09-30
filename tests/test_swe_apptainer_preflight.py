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
    setup.write_text("""module() {
    [[ "$PROBE_FAILURE" != module ]] || return 30
    : "$BASH_COMPLETION_DEBUG"
}
uname() { echo aarch64; }
hostname() { if [[ "$PROBE_FAILURE" == login ]]; then echo login1.vista.tacc.utexas.edu; else echo c609-071; fi; }
flock() { :; }
apptainer() {
    [[ "$-" == *u* ]] || return 97
    case "$1" in
        --version) if [[ "$PROBE_FAILURE" == wrapper ]]; then echo 'Please do not run Apptainer on login nodes!'; else echo 'apptainer version 1.4.1'; fi;;
        build) [[ "$PROBE_FAILURE" != pull ]] || return 31;
            [[ "$2" == --mksquashfs-args && "$3" == '-processors 1 -mem 256M' && "$GOMAXPROCS" == 1 ]] || return 98;
            [[ "$PROBE_FAILURE" != missing_image ]] || return 0;
            printf 'mock-sif' > "$4";;
        exec) [[ "$PROBE_FAILURE" != tests ]] || return 32;
            [[ "$PROBE_FAILURE" != no_tests ]] || return 0;
            echo 'SWE_APPTAINER_PUBLIC_TESTS_PASSED';;
        *) return 99;;
    esac
}
""", encoding="utf-8", newline="\n")
    env = dict(os.environ, BASH_ENV=setup.as_posix(), PROJECT_ROOT=ROOT.as_posix(),
               SCRATCH=scratch.as_posix(), PROBE_FAILURE=failure)
    env.pop("SLURM_JOB_ID", None)
    env.pop("BASH_COMPLETION_DEBUG", None)
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


@pytest.mark.parametrize("failure,code,stage", [("login", 2, "environment"), ("wrapper", 2, "environment"), ("module", 30, "environment"), ("pull", 31, "image_pull"), ("missing_image", 2, "image_pull"), ("tests", 32, "repository_and_tests"), ("no_tests", 2, "repository_and_tests")])
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


def test_direct_run_has_separate_artifacts_and_shell_syntax_is_valid(tmp_path):
    result = run_batch(tmp_path, job=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "SWE_APPTAINER_PREFLIGHT_COMPLETE" in result.stdout
    assert "run=direct" in result.stdout
    assert len(list(tmp_path.glob("scratch/context-graph-swe/runs/preflight-direct-*/suite.log"))) == 1
    for path in (BATCH, PROBE):
        result = subprocess.run([BASH, "-n", path.as_posix()], capture_output=True,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        assert result.returncode == 0, result.stderr
