import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import builtins
from types import SimpleNamespace

import pytest

from scripts.eval_swebench_verified import DATASET, file_hash, write_json
from scripts.prepare_swe_verified_vista import REVISION, check_data


def test_prepare_checks_metadata_without_importing_model_runtimes(tmp_path, monkeypatch):
    from scripts import prepare_swe_verified_vista as setup
    prepared(tmp_path)
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "ray", "tensordict", "tensorflow", "jax"}:
            raise AssertionError(f"Unnecessary runtime import: {name}")
        return original_import(name, *args, **kwargs)

    def tokenizer_load(*args, **kwargs):
        assert all(os.environ[name] == "0" for name in ("USE_TORCH", "USE_TF", "USE_FLAX"))
        return SimpleNamespace(apply_chat_template=lambda *a, **k: [1, 2])

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(setup.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(setup.importlib.metadata, "version", lambda name: "test-version")
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer_load)))
    for name in ("USE_TORCH", "USE_TF", "USE_FLAX"):
        monkeypatch.setenv(name, "1")
    report = tmp_path / "environment.json"
    monkeypatch.setattr(sys, "argv", ["prepare", "--data-dir", str(tmp_path),
                                    "--model-path", str(tmp_path), "--report", str(report)])
    setup.main()
    result = json.loads(report.read_text())
    assert result["versions"]["transformers"] == "test-version"
    assert "torch" not in result["versions"]
    assert result["runtime_imports_checked"] is False
    assert result["evaluation_performed"] is False


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
@pytest.mark.parametrize("lock_status,gpu_pids,gpu_status,expected", [
    (1, "", 0, "SWE_SERVER_ALREADY_RUNNING"),
    (0, "571393\n571401", 0, "SWE_SERVER_GPU_BUSY"),
    (0, "", 1, ""),
    (0, "", 0, ""),
])
def test_swe_server_checks_lock_and_gpu_before_model_setup(tmp_path, lock_status, gpu_pids, gpu_status, expected):
    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts/serve_swe_qwen35_9b_vista.sbatch").read_text()
    script = tmp_path / "server.sh"
    script.write_text(source.replace('/tmp/contextgraph-swe-server-${UID}.lock',
                                     (tmp_path / "server.lock").as_posix()), newline="\n")
    mocks = tmp_path / "mocks.sh"
    mocks.write_text('flock() { return "$TEST_LOCK_STATUS"; }\n'
                     'nvidia-smi() { printf "%s" "$TEST_GPU_PIDS"; return "$TEST_GPU_STATUS"; }\n', newline="\n")
    conda = tmp_path / "conda.sh"
    conda.write_text('touch "$PROJECT_ROOT/model-setup-reached"\nexit 0\n', newline="\n")
    env = dict(os.environ, PROJECT_ROOT=tmp_path.as_posix(), SCRATCH=tmp_path.as_posix(),
               SLURM_JOB_ID="123", CONDA_SH=conda.as_posix(), BASH_ENV=mocks.as_posix(),
               TEST_LOCK_STATUS=str(lock_status), TEST_GPU_PIDS=gpu_pids, TEST_GPU_STATUS=str(gpu_status))
    result = subprocess.run([shutil.which("bash"), script.as_posix()], env=env, text=True,
                            capture_output=True, timeout=10,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    allowed = lock_status == gpu_status == 0 and not gpu_pids
    assert (result.returncode == 0) == allowed, result.stdout + result.stderr
    assert (tmp_path / "model-setup-reached").exists() == allowed
    assert expected in result.stderr


@pytest.mark.skipif(not shutil.which("bash"), reason="Bash required")
@pytest.mark.parametrize("allocated", [False, True])
@pytest.mark.parametrize("reuse", [False, True])
def test_direct_setup_marks_image_pending_and_allocation_runs_probe(tmp_path, allocated, reuse):
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
    local destination="${@: -1}"
    mkdir -p "$destination/bin"
    printf '#!/bin/bash\\nexit 0\\n' > "$destination/bin/python"
    chmod +x "$destination/bin/python"
}
''', encoding="utf8", newline="\n")
    env = dict(os.environ, PROJECT_ROOT=tmp_path.as_posix(), SCRATCH=scratch.as_posix(),
               BASH_ENV=mocks.as_posix(), SWE_BASE_PYTHON="fake_python")
    env.pop("SLURM_JOB_ID", None)
    env.pop("SWE_AGENT_ENV", None)
    if reuse:
        venv = scratch / "context-graph-swe/envs/previous"
        (venv / "bin").mkdir(parents=True)
        (venv / "pyvenv.cfg").write_text("include-system-site-packages = true\n")
        (venv / "bin/python").write_text("#!/bin/bash\nexit 0\n", newline="\n")
        (venv / "bin/python").chmod(0o755)
        env["SWE_AGENT_ENV"] = venv.as_posix()
    if allocated:
        env["SLURM_JOB_ID"] = "123"
    result = subprocess.run([shutil.which("bash"), (root / "scripts/setup_swe_verified_vista.sbatch").as_posix()],
                            env=env, text=True, capture_output=True, timeout=30,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    assert result.returncode == 0, result.stdout + result.stderr
    if reuse:
        assert list((scratch / "context-graph-swe/envs").iterdir()) == [venv]
    assert (tmp_path / "probe-ran").exists() == allocated
    completion = next(scratch.glob("context-graph-swe/runs/*/setup-complete.txt")).read_text()
    assert f"image_preflight={'passed' if allocated else 'pending'}" in completion
    assert "evaluation_performed=false" in completion
    assert ("SWE_VISTA_SETUP_COMPLETE" in result.stdout) == allocated
    assert ("SWE_VISTA_PYTHON_DATA_READY" in result.stdout) != allocated
