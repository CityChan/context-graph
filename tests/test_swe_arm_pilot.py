import hashlib
from pathlib import Path
from types import SimpleNamespace
import os
import shutil
import subprocess
import sys

import pytest

from envs.swebench_apptainer import ApptainerSandbox, BASE, IMAGE_NAME, INSTANCE, checked_image


def task():
    return dict(instance_id=INSTANCE, repo="sympy/sympy", base_commit=BASE, problem_statement="Fix the bug")


def cached(root):
    image = root / "images" / IMAGE_NAME
    image.parent.mkdir(parents=True)
    image.write_bytes(b"fixture-sif")
    sha = hashlib.sha256(image.read_bytes()).hexdigest()
    Path(str(image) + ".sha256").write_text(f"{sha}  {IMAGE_NAME}\n")
    return image


def test_pilot_rejects_other_instances_and_changed_image(tmp_path):
    image = cached(tmp_path)
    assert checked_image(tmp_path, task())[0] == image
    with pytest.raises(ValueError, match="only the pinned"):
        checked_image(tmp_path, dict(task(), instance_id="sympy__sympy-12345"))
    image.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        checked_image(tmp_path, task())


def test_exec_masks_original_testbed_and_drops_host_secrets(tmp_path, monkeypatch):
    sandbox = ApptainerSandbox(task(), root=tmp_path)
    sandbox.image = cached(tmp_path)
    sandbox.work = tmp_path / "sandboxes/sympy-test"
    sandbox.work.mkdir(parents=True)
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        kwargs["stdout"].write(b"abcdef")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret")
    monkeypatch.setenv("APPTAINER_BIND", "/home:/secrets")
    monkeypatch.setenv("LD_PRELOAD", "/host/torch.so")
    status, output = sandbox._exec("echo hello", limit=3)
    argv, kw = calls[0]
    assert status == 0 and output == "abc\n[output truncated]"
    assert argv[:4] == ["timeout", "--signal=TERM", "--kill-after=5", "90"]
    assert argv[argv.index("--network") + 1] == "none"
    assert f"{sandbox.work}:/testbed" in argv
    assert "--containall" in argv and "--no-home" in argv
    assert not {"OPENAI_API_KEY", "APPTAINER_BIND", "LD_PRELOAD"}.intersection(kw["env"])
    sandbox.close()
    assert not (tmp_path / "sandboxes/sympy-test").exists()


def test_start_cleans_failed_worktree_and_patch_failure_is_not_empty_success(tmp_path, monkeypatch):
    cached(tmp_path)
    monkeypatch.setattr("envs.swebench_apptainer.platform.system", lambda: "Linux")
    monkeypatch.setattr("envs.swebench_apptainer.platform.machine", lambda: "aarch64")
    monkeypatch.setattr("envs.swebench_apptainer.platform.node", lambda: "c612-041")
    sandbox = ApptainerSandbox(task(), root=tmp_path)
    monkeypatch.setattr(sandbox, "_exec", lambda *a, **kw: (1, "namespace unavailable"))
    with pytest.raises(RuntimeError, match="initialization failed"):
        sandbox.start()
    with pytest.raises(RuntimeError, match="Patch extraction|patch extraction"):
        sandbox.patch()
    sandbox.close()
    assert not list((tmp_path / "sandboxes").iterdir())


@pytest.mark.skipif(not shutil.which("bash"), reason="Bash required")
def test_runner_shell_syntax():
    runner = Path(__file__).resolve().parents[1] / "scripts/run_swe_arm_pilot_idev.sh"
    result = subprocess.run([shutil.which("bash"), "-n", runner.as_posix()], capture_output=True,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("mode", ["pass", "fail", "apply_failed", "wrapper", "timeout", "missing_tests"])
def test_grading_distinguishes_model_failure_from_infrastructure(tmp_path, monkeypatch, mode):
    from scripts import grade_swe_arm_pilot as grade
    spec = SimpleNamespace(eval_script="upstream test script", FAIL_TO_PASS=["test_bug"], PASS_TO_PASS=[])
    monkeypatch.setitem(sys.modules, "swebench.harness.test_spec.test_spec",
                        SimpleNamespace(make_test_spec=lambda task: spec))
    monkeypatch.setitem(sys.modules, "swebench.harness.grading", SimpleNamespace(
        get_logs_eval=lambda *a: ({"test_bug": "PASSED"}, mode != "missing_tests"),
        get_eval_report=lambda *a, **k: {INSTANCE: {"resolved": mode == "pass"}}))

    class Sandbox:
        closed = False
        provenance = {"sif_sha256": "fixture"}
        def __init__(self, *a, **k): pass
        def start(self): pass
        def _exec(self, command, **kwargs):
            if "git apply" in command:
                return (1 if mode == "apply_failed" else 0,
                        "site refused execution" if mode == "wrapper" else "SWE_ARM_APPLY_STARTED")
            assert kwargs["inputs"] == tmp_path / "grade/inputs"
            return (124 if mode == "timeout" else 0), "test log"
        def close(self): Sandbox.closed = True

    monkeypatch.setattr(grade, "ApptainerSandbox", Sandbox)
    if mode in {"wrapper", "timeout", "missing_tests"}:
        with pytest.raises(RuntimeError):
            grade.evaluate(task(), "patch", tmp_path / "grade", tmp_path, 10)
        assert not (tmp_path / "grade/report.json").exists()
    else:
        report = grade.evaluate(task(), "patch", tmp_path / "grade", tmp_path, 10)
        assert report[INSTANCE]["resolved"] == (mode == "pass")
    assert Sandbox.closed
