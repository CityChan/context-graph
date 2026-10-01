import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import os
import shutil
import subprocess
import sys

import pytest

from envs.swebench_apptainer import ApptainerSandbox, BASE, IMAGE_NAME, INSTANCE, checked_image
from scripts.eval_swebench_verified import DATASETS, file_hash, write_json
from scripts.grade_swe_arm_pilot import load_task


def task():
    return dict(instance_id=INSTANCE, repo="sympy/sympy", base_commit=BASE, problem_statement="Fix the bug")


def test_lite_arm_pilot_uses_pinned_sympy_task_and_rejects_verified_data(tmp_path):
    data = tmp_path / "lite"
    (data / "public").mkdir(parents=True)
    (data / "grading").mkdir()
    rows = [dict(task(), patch="gold", test_patch="test")]
    rows += [dict(instance_id=f"django__django-{i}", repo="django/django", base_commit="a" * 40,
                  problem_statement="Fix a bug", patch="gold", test_patch="test") for i in range(299)]
    write_json(data / "public/instances.json", [
        {key: row[key] for key in ("instance_id", "repo", "base_commit", "problem_statement")} for row in rows])
    write_json(data / "grading/instances.json", rows)
    write_json(data / "manifest.json", {"dataset": DATASETS["lite"][0], "split": "test", "count": 300,
        "revision": "a" * 40, "public_sha256": file_hash(data / "public/instances.json"),
        "grading_sha256": file_hash(data / "grading/instances.json")})
    selected, manifest = load_task(data, "lite")
    assert selected["instance_id"] == INSTANCE and manifest["dataset"] == DATASETS["lite"][0]
    with pytest.raises(ValueError, match="requested benchmark"):
        load_task(data, "verified")


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


@pytest.mark.parametrize("patch", [b"", b"diff --git a/file b/file\n--- a/file\n+++ b/file\n"])
def test_patch_excludes_container_stderr(tmp_path, monkeypatch, patch):
    sandbox = ApptainerSandbox(task(), root=tmp_path)
    sandbox.image = cached(tmp_path)
    sandbox.work = tmp_path

    def run(argv, **kwargs):
        kwargs["stdout"].write(patch)
        kwargs["stderr"].write(b"INFO: gocryptfs not found, will not be able to use gocryptfs\n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", run)
    assert sandbox.patch() == patch.decode()


def test_patch_preserves_failed_command_diagnostics(tmp_path, monkeypatch):
    sandbox = ApptainerSandbox(task(), root=tmp_path)
    sandbox.image = cached(tmp_path)
    sandbox.work = tmp_path

    def run(argv, **kwargs):
        kwargs["stderr"].write(b"fatal: bad revision")
        return SimpleNamespace(returncode=128)

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(RuntimeError, match="fatal: bad revision"):
        sandbox.patch()


def test_private_grading_environment_mount_and_cleanup(tmp_path, monkeypatch):
    sandbox = ApptainerSandbox(task(), root=tmp_path)
    sandbox.image = cached(tmp_path)
    sandbox.work = tmp_path / "sandboxes/sympy-test"
    sandbox.work.mkdir(parents=True)
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", run)
    sandbox.prepare_grading_environment()
    runtime = sandbox.runtime
    assert f"{runtime}:/runtime-copy" in calls[0]
    assert "cp -a --no-preserve=ownership /opt/miniconda3/envs/testbed/." in calls[0][-1]
    sandbox._exec("true")
    assert f"{runtime}:/opt/miniconda3/envs/testbed" in calls[1]
    sandbox.close()
    assert not runtime.exists()


@pytest.mark.skipif(not shutil.which("bash"), reason="Bash required")
@pytest.mark.parametrize("install_exit", [0, 1])
def test_setup_failure_stops_but_test_failure_keeps_end_marker(tmp_path, install_exit):
    from scripts.grade_swe_arm_pilot import checked_eval_script
    script = "set -uxo pipefail\npython -m pip install -e .\n: '>>>>> Start Test Output'\nfalse\necho TESTS_COMPLETED\n"
    # Shell fixture: stand in for pip/import, exercising the actual error boundaries.
    prefix = f"python() {{ return {install_exit}; }}\n"
    target = tmp_path / "eval.sh"
    target.write_text(prefix + checked_eval_script(script), newline="\n")
    result = subprocess.run([shutil.which("bash"), target.as_posix()], capture_output=True,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    assert (b"TESTS_COMPLETED" in result.stdout) == (install_exit == 0)
    assert result.returncode == install_exit


@pytest.mark.skipif(not shutil.which("bash"), reason="Bash required")
def test_grading_git_config_uses_private_home(tmp_path):
    from scripts.grade_swe_arm_pilot import checked_eval_script
    script = ("set -uxo pipefail\n"
              "git config --global --add safe.directory /testbed\n"
              "git config --global --get-all safe.directory\n"
              "python -m pip install -e .\n"
              ": '>>>>> Start Test Output'\ntrue\n")
    # Run real Git with a nonexistent inherited HOME. Replace only allocation of
    # the private container /tmp directory and the unavailable container Python.
    private = tmp_path / "private-home"
    private.mkdir()
    target = tmp_path / "eval.sh"
    target.write_text('mktemp() { printf "%s" "$TEST_PRIVATE_HOME"; }\npython() { return 0; }\n'
                      + checked_eval_script(script), newline="\n")
    env = dict(os.environ, HOME=str(tmp_path / "missing-host-home"),
               XDG_CONFIG_HOME=str(tmp_path / "missing-host-config"),
               TEST_PRIVATE_HOME=private.as_posix())
    env.pop("GIT_CONFIG_GLOBAL", None)
    result = subprocess.run([shutil.which("bash"), target.as_posix()], env=env, capture_output=True,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    assert b"/testbed" in result.stdout
    assert "/testbed" in (private / ".gitconfig").read_text()
    assert not (tmp_path / "missing-host-home").exists()


@pytest.mark.skipif(not shutil.which("bash"), reason="Bash required")
def test_runner_shell_syntax():
    runner = Path(__file__).resolve().parents[1] / "scripts/run_swe_arm_pilot_idev.sh"
    result = subprocess.run([shutil.which("bash"), "-n", runner.as_posix()], capture_output=True,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("mode", ["pass", "fail", "apply_failed", "wrapper", "timeout", "missing_tests", "setup_failed", "missing_import"])
def test_grading_distinguishes_model_failure_from_infrastructure(tmp_path, monkeypatch, mode):
    from scripts import grade_swe_arm_pilot as grade
    spec = SimpleNamespace(eval_script="set -uxo pipefail\npython -m pip install -e .\n: '>>>>> Start Test Output'\nfalse\n", FAIL_TO_PASS=["test_bug"], PASS_TO_PASS=[])
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
        def prepare_grading_environment(self): pass
        def _exec(self, command, **kwargs):
            if "git apply" in command:
                return (1 if mode == "apply_failed" else 0,
                        "site refused execution" if mode == "wrapper" else "SWE_ARM_APPLY_STARTED")
            assert kwargs["inputs"] == tmp_path / "grade/inputs"
            return (124 if mode == "timeout" else 1 if mode == "setup_failed" else 0), ("test log" if mode == "missing_import" else "SWE_ARM_IMPORT_OK /testbed/sympy/__init__.py\ntest log")
        def close(self): Sandbox.closed = True

    monkeypatch.setattr(grade, "ApptainerSandbox", Sandbox)
    if mode in {"wrapper", "timeout", "missing_tests", "setup_failed", "missing_import"}:
        with pytest.raises(RuntimeError):
            grade.evaluate(task(), "patch", tmp_path / "grade", tmp_path, 10)
        assert not (tmp_path / "grade/report.json").exists()
    else:
        report = grade.evaluate(task(), "patch", tmp_path / "grade", tmp_path, 10)
        assert report[INSTANCE]["resolved"] == (mode == "pass")
    assert Sandbox.closed
