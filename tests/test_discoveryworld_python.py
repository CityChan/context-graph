import json
import subprocess
import sys

import pytest

from scripts import select_discoveryworld_python as selector


@pytest.mark.parametrize("version", [[3, 9], [3, 12], [3, 13]])
def test_reject_incompatible_python_before_install(monkeypatch, version):
    monkeypatch.setattr(selector.subprocess, "run", lambda *a, **k:
                        subprocess.CompletedProcess(a, 0, json.dumps({"version": version, "missing": []})))
    with pytest.raises(RuntimeError, match="requires 3.10/3.11"):
        selector.check_python("cached-overlay/bin/python")


def test_reject_bare_python_without_agent_dependencies(monkeypatch):
    monkeypatch.setattr(selector.subprocess, "run", lambda *a, **k:
                        subprocess.CompletedProcess(a, 0, json.dumps({"version": [3, 11], "missing": ["torch"]})))
    with pytest.raises(RuntimeError, match="missing agent packages: torch"):
        selector.check_python("python")


def test_select_skips_incompatible_candidate_and_preserves_venv_path(monkeypatch, tmp_path):
    bad, good = tmp_path / "old-python", tmp_path / "venv-python"
    bad.touch()
    good.touch()
    monkeypatch.setattr(selector, "candidates", lambda env: [str(bad), str(good)])
    visited = []

    def check(path):
        visited.append(path)
        if path == str(bad):
            raise RuntimeError("Python 3.12")
        return {"version": [3, 11], "missing": []}

    monkeypatch.setattr(selector, "check_python", check)
    assert selector.select_python({}) == str(good)
    assert visited == [str(bad), str(good)]


def test_explicit_incompatible_override_is_not_silently_replaced(monkeypatch):
    def check(path):
        raise RuntimeError("explicit Python 3.12")

    monkeypatch.setattr(selector, "check_python", check)
    monkeypatch.setattr(selector, "candidates", lambda env: pytest.fail("Must not fall back"))
    with pytest.raises(RuntimeError, match="explicit Python 3.12"):
        selector.select_python({"BENCH_BASE_PYTHON": "requested-python"})


def test_no_compatible_candidate_reports_inspected_paths(monkeypatch, tmp_path):
    path = tmp_path / "python"
    path.touch()
    monkeypatch.setattr(selector, "candidates", lambda env: [str(path)])

    def check(path):
        raise RuntimeError(f"{path}: Python 3.12")

    monkeypatch.setattr(selector, "check_python", check)
    with pytest.raises(RuntimeError, match="BENCH_BASE_PYTHON") as error:
        selector.select_python({})
    assert str(path) in str(error.value)


def test_probe_real_interpreter():
    if sys.version_info[:2] not in ((3, 10), (3, 11)):
        pytest.skip("Live check requires a compatible agent test environment")
    # Developer test environments may rely on user-site packages. The production
    # launcher disables those, so the isolated probe must report that dependency gap.
    import os
    probe = subprocess.run([sys.executable, "-I", "-c", selector.PROBE],
                           capture_output=True, text=True, check=True,
                           creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    info = json.loads(probe.stdout)
    assert info["version"] == list(sys.version_info[:2])
    if info["missing"]:
        with pytest.raises(RuntimeError, match="missing agent packages"):
            selector.check_python(sys.executable)
    else:
        assert selector.check_python(sys.executable) == info
