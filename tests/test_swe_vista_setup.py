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
