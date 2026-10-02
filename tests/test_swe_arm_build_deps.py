from pathlib import Path
from types import SimpleNamespace
import subprocess

import pytest

from envs.swebench_apptainer import ApptainerSandbox, BASE, INSTANCE
from scripts import grade_swe_arm_pilot as grade
from scripts import prepare_swe_arm_build_deps as prepare


def task():
    return dict(instance_id=INSTANCE, base_commit=BASE, repo="sympy/sympy", problem_statement="Fix")


def test_reads_all_declared_build_requirements_without_executing_project(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        '[build-system]\nrequires = ["setuptools", "setuptools_scm>=6.2", "wheel", '
        '"cython==0.29.22", "oldest-supported-numpy", "extension-helpers"]\n')
    (tmp_path / "setup.py").write_text('raise RuntimeError("project setup must not run")')
    assert prepare.build_requirements(tmp_path) == ["setuptools", "setuptools_scm>=6.2", "wheel",
                                                   "cython==0.29.22", "oldest-supported-numpy", "extension-helpers"]
    (tmp_path / "pyproject.toml").write_text('[build-system]\nrequires = ["--force-reinstall"]\n')
    with pytest.raises(ValueError, match="Invalid build-system"):
        prepare.build_requirements(tmp_path)


def test_preparation_installs_declared_dependencies_and_records_versions(monkeypatch, capsys):
    monkeypatch.setattr(prepare.sys, "prefix", "/opt/miniconda3/envs/testbed")
    monkeypatch.setattr(prepare, "build_requirements", lambda path: ["extension-helpers", "cython==0.29.22"])
    calls = []
    monkeypatch.setattr(prepare.subprocess, "check_call", lambda args: calls.append(args))
    for key in ("PIP_NO_INDEX", "PIP_NO_DEPS", "PIP_NO_BUILD_ISOLATION"):
        monkeypatch.setenv(key, "1")
    prepare.main()
    assert [args[3:] for args in calls] == [["freeze", "--all"],
        ["install", "extension-helpers", "cython==0.29.22"], ["freeze", "--all"]]
    assert "SWE_ARM_BUILD_ENV_READY" in capsys.readouterr().out


def test_network_exception_is_per_call_and_does_not_reach_agent_tools(tmp_path, monkeypatch):
    sandbox = ApptainerSandbox(task(), root=tmp_path)
    sandbox.work = tmp_path
    sandbox.image = tmp_path / "image.sif"
    calls = []
    def run(argv, **kwargs):
        calls.append((argv, kwargs["env"]))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setenv("APPTAINER_BIND", "/host:/host")
    sandbox._exec("trusted dependency setup", trusted_setup_network=True)
    sandbox.execute("print('agent')")
    sandbox._exec("/bin/bash /grading-input/eval.sh")
    assert "--net" not in calls[0][0]
    for argv, _ in calls[1:]:
        assert argv[argv.index("--network") + 1] == "none"
    for _, env in calls:
        assert "OPENAI_API_KEY" not in env and "APPTAINER_BIND" not in env


def test_runtime_template_is_copied_not_shared(tmp_path):
    template = tmp_path / "template"
    template.mkdir()
    (template / "package").write_text("prepared")
    sandbox = ApptainerSandbox(task(), root=tmp_path)
    sandbox.work = tmp_path / "sandboxes/sympy-copy"
    sandbox.work.mkdir(parents=True)
    sandbox.prepare_grading_environment(template)
    (sandbox.runtime / "package").write_text("changed by project install")
    assert (template / "package").read_text() == "prepared"
    runtime = sandbox.runtime
    sandbox.close()
    assert not runtime.exists() and template.exists()


@pytest.mark.parametrize("failed", [False, True])
def test_preparation_transfers_only_successful_environment_and_cleans_failures(tmp_path, monkeypatch, failed):
    created = []
    class Sandbox:
        provenance = {"sif_sha256": "sha"}
        runtime_ready = True
        closed = False
        def __init__(self, *a, **kw):
            self.runtime = tmp_path / "runtime-original"
            self.runtime.mkdir()
            created.append(self)
        def start(self): pass
        def prepare_grading_environment(self): pass
        def _exec(self, command, **kw):
            assert kw["trusted_setup_network"] is True
            assert sorted(p.name for p in kw["inputs"].iterdir()) == ["prepare.py"]
            assert "git apply" not in command
            return (1, "missing build tool") if failed else (0, "SWE_ARM_BUILD_ENV_READY\n")
        def close(self):
            self.closed = True
            if self.runtime is not None:
                self.runtime.rmdir()
    monkeypatch.setattr(grade, "ApptainerSandbox", Sandbox)
    folder = tmp_path / "build-environment"
    if failed:
        with pytest.raises(RuntimeError, match="Build dependency preparation failed"):
            grade.prepare_build_environment(task(), folder, tmp_path, 60)
        assert not (folder / "runtime").exists()
    else:
        assert grade.prepare_build_environment(task(), folder, tmp_path, 60) == folder / "runtime"
        assert (folder / "environment.json").exists()
    assert created[0].closed and not (tmp_path / "runtime-original").exists()
    assert (folder / "prepare.log").exists()
