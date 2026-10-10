from pathlib import Path
from types import SimpleNamespace
import os
import shutil
import subprocess

import pytest

from scripts import prepare_amem_environment as setup


def test_missing_dependency_installs_only_in_unique_venv(monkeypatch, tmp_path):
    monkeypatch.setattr(setup.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(setup, "constraints", lambda: "torch==2.6.0\ntransformers==5.3.0\n")
    calls = []
    monkeypatch.setattr(setup, "run", calls.append)
    first = setup.prepare(tmp_path)
    second = setup.prepare(tmp_path)
    assert first != second
    assert calls[0][:4] == [setup.sys.executable, "-m", "venv", "--system-site-packages"]
    assert calls[1][0] == first
    assert calls[1][1:5] == ["-m", "pip", "--isolated", "install"]
    assert "--only-binary=:all:" in calls[1]
    assert calls[1][-1] == "sentence-transformers==5.3.0"
    constraints = Path(calls[1][calls[1].index("--constraint") + 1])
    assert "torch==2.6.0" in constraints.read_text()
    assert calls[2][0] == first  # Verify the exact interpreter used by evaluators.


def test_existing_dependency_is_probed_without_install(monkeypatch, tmp_path):
    monkeypatch.setattr(setup.importlib.util, "find_spec", lambda name: object())
    calls = []
    monkeypatch.setattr(setup, "run", calls.append)
    assert setup.prepare(tmp_path) == str(Path(setup.sys.executable).absolute())
    assert len(calls) == 1 and calls[0][1] == "-c"
    assert not list(tmp_path.iterdir())


def test_constraints_preserve_first_visible_versions(monkeypatch):
    packages = [SimpleNamespace(metadata={"Name": name}, version=version) for name, version in
                [("torch", "2.6.0+cu124"), ("torch", "2.5.0"), ("typing_extensions", "4.12.0"),
                 ("sentence-transformers", "1.0.0"), ("", "0")]]
    monkeypatch.setattr(setup.importlib.metadata, "distributions", lambda: packages)
    assert setup.constraints() == "torch==2.6.0+cu124\ntyping-extensions==4.12.0\n"


def test_setup_failure_never_publishes_interpreter(monkeypatch, tmp_path):
    marker = tmp_path / "amem-python.txt"
    monkeypatch.setattr(setup.sys, "argv", ["setup", "--root", str(tmp_path), "--python-file", str(marker)])
    def fail(root):
        raise RuntimeError("dependency conflict")
    monkeypatch.setattr(setup, "prepare", fail)
    with pytest.raises(RuntimeError, match="dependency conflict"):
        setup.main()
    assert not marker.exists()


def test_broken_existing_install_does_not_trigger_reinstallation(monkeypatch, tmp_path):
    monkeypatch.setattr(setup.importlib.util, "find_spec", lambda name: object())
    def fail(args):
        raise RuntimeError("import error")
    monkeypatch.setattr(setup, "run", fail)
    with pytest.raises(RuntimeError, match="import error"):
        setup.prepare(tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.skipif(not shutil.which("bash"), reason="Bash required")
@pytest.mark.parametrize("method", ["amem", "memobrain", "supo"])
def test_evaluator_uses_amem_interpreter_only_for_amem(tmp_path, method):
    root = Path(__file__).resolve().parents[1]
    for name in ("python", "amem-python"):
        stub = tmp_path / name
        stub.write_text('#!/bin/bash\nprintf "%s %s\\n" "${0##*/}" "$*" >> "$PROJECT_ROOT/calls.txt"\n', newline="\n")
        stub.chmod(0o755)
    conda = tmp_path / "conda.sh"
    conda.write_text('conda() { export PATH="$(pwd):$PATH"; }\n', newline="\n")
    env = dict(os.environ, PROJECT_ROOT=tmp_path.as_posix(), CONDA_SH=conda.as_posix(),
               METHOD=method, SCRATCH=tmp_path.as_posix(), MODEL_PATH="fixture-model",
               RUN_ROOT="fixture-run", AMEM_AGENT_PYTHON=(tmp_path / "amem-python").as_posix())
    proc = subprocess.run([shutil.which("bash"), (root / "scripts/eval_bcp_qwen38_4node_idev.sh").as_posix(),
                           "_eval", "http://fixture", "1"], env=env, capture_output=True, text=True,
                          timeout=15, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    calls = (tmp_path / "calls.txt").read_text().splitlines()
    if method == "amem":
        assert len(calls) == 2
        assert calls[0] == "amem-python scripts/prepare_amem_embedding.py --offline"
        assert calls[1].startswith("amem-python -u scripts/eval_bcp_qwen38.py ")
    else:
        assert len(calls) == 1
        assert calls[0].startswith("python -u scripts/eval_bcp_qwen38.py ")
