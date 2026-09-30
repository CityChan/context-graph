import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from scripts.eval_swebench_verified import DATASET, file_hash, write_json
from scripts.prepare_swe_verified_vista import REVISION, check_data


def prepared(root):
    (root / "public").mkdir()
    (root / "grading").mkdir()
    rows = [dict(instance_id=f"sympy__sympy-{i}", repo="sympy/sympy",
                 base_commit="a" * 40, problem_statement="Fix issue") for i in range(499)]
    rows.append(dict(instance_id="sympy__sympy-20590", repo="sympy/sympy",
                     base_commit="cffd4e0f86fefd4802349a9f9b19ed70934ea354", problem_statement="Fix issue"))
    write_json(root / "public/instances.json", rows)
    write_json(root / "grading/instances.json", [dict(row, patch="gold") for row in rows])
    manifest = dict(dataset=DATASET, revision=REVISION, split="test", count=500,
                    public_sha256=file_hash(root / "public/instances.json"),
                    grading_sha256=file_hash(root / "grading/instances.json"))
    write_json(root / "manifest.json", manifest)
    return manifest


def test_reuse_validates_both_dataset_files_and_revision(tmp_path):
    manifest = prepared(tmp_path)
    assert check_data(tmp_path) == manifest
    assert check_data(tmp_path) == manifest
    (tmp_path / "grading/instances.json").write_text("[]")
    with pytest.raises(ValueError, match="Grading dataset hash"):
        check_data(tmp_path)


def test_reject_different_revision_or_probe_commit(tmp_path):
    manifest = prepared(tmp_path)
    manifest["revision"] = "b" * 40
    write_json(tmp_path / "manifest.json", manifest)
    with pytest.raises(ValueError, match="revision"):
        check_data(tmp_path)
    manifest["revision"] = REVISION
    rows = json.loads((tmp_path / "public/instances.json").read_text())
    rows[-1]["base_commit"] = "b" * 40
    write_json(tmp_path / "public/instances.json", rows)
    manifest["public_sha256"] = file_hash(tmp_path / "public/instances.json")
    write_json(tmp_path / "manifest.json", manifest)
    with pytest.raises(ValueError, match="base commit"):
        check_data(tmp_path)


@pytest.mark.skipif(not shutil.which("bash"), reason="Bash required")
def test_setup_shell_syntax():
    script = Path(__file__).resolve().parents[1] / "scripts/setup_swe_verified_vista.sbatch"
    result = subprocess.run([shutil.which("bash"), "-n", script.as_posix()], capture_output=True,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(not shutil.which("bash"), reason="Bash required")
@pytest.mark.parametrize("allocated", [False, True])
def test_direct_setup_marks_image_pending_and_allocation_runs_probe(tmp_path, allocated):
    root = Path(__file__).resolve().parents[1]
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/preflight_swe_apptainer_vista.sbatch").write_text(
        'touch "$PROJECT_ROOT/probe-ran"\n', encoding="utf8", newline="\n")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    mocks = tmp_path / "mocks.sh"
    mocks.write_text('''uname() { echo aarch64; }
git() { echo fixture-commit; }
flock() { :; }
fake_python() {
    mkdir -p "$4/bin"
    printf '#!/bin/bash\\nexit 0\\n' > "$4/bin/python"
    chmod +x "$4/bin/python"
}
''', encoding="utf8", newline="\n")
    env = dict(os.environ, PROJECT_ROOT=tmp_path.as_posix(), SCRATCH=scratch.as_posix(),
               BASH_ENV=mocks.as_posix(), SWE_BASE_PYTHON="fake_python")
    env.pop("SLURM_JOB_ID", None)
    if allocated:
        env["SLURM_JOB_ID"] = "123"
    result = subprocess.run([shutil.which("bash"), (root / "scripts/setup_swe_verified_vista.sbatch").as_posix()],
                            env=env, text=True, capture_output=True, timeout=30,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "probe-ran").exists() == allocated
    completion = next(scratch.glob("context-graph-swe/runs/*/setup-complete.txt")).read_text()
    assert f"image_preflight={'passed' if allocated else 'pending'}" in completion
    assert "evaluation_performed=false" in completion
    assert ("SWE_VISTA_SETUP_COMPLETE" in result.stdout) == allocated
    assert ("SWE_VISTA_PYTHON_DATA_READY" in result.stdout) != allocated
