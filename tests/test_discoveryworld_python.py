import os
import subprocess
import sys

import pytest

from scripts import select_discoveryworld_python as selector


@pytest.mark.parametrize("version", [[3, 9], [3, 12], [3, 13]])
def test_reject_incompatible_python_before_install(monkeypatch, version):
    monkeypatch.setattr(selector.subprocess, "run", lambda *a, **k:
                        subprocess.CompletedProcess(a, 0, ".".join(map(str, version))))
    with pytest.raises(RuntimeError, match="requires 3.10/3.11"):
        selector.check_python("cached-overlay/bin/python")


def test_version_check_disables_site_initialization(monkeypatch):
    def run(argv, **kwargs):
        assert argv[1:4] == ["-I", "-S", "-c"]
        assert "find_spec" not in argv[-1]
        return subprocess.CompletedProcess(argv, 0, "3.10\n")

    monkeypatch.setattr(selector.subprocess, "run", run)
    assert selector.check_python("python") == {"version": [3, 10]}


def test_timeout_reports_unknown_compatibility(monkeypatch):
    def run(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, 30)

    monkeypatch.setattr(selector.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="compatibility is unknown"):
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
        return {"version": [3, 11]}

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
    assert selector.check_python(sys.executable) == {"version": list(sys.version_info[:2])}


def test_version_probe_bypasses_site_hooks_in_real_venv(tmp_path):
    import venv

    root = tmp_path / "agent"
    venv.EnvBuilder(with_pip=False).create(root)
    if os.name == "nt":
        python = root / "Scripts/python.exe"
        site = root / "Lib/site-packages"
    else:
        python = root / "bin/python"
        site = root / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
    marker = tmp_path / "site-hook-ran"
    # A marker proves the hook runs under -I, but is bypassed by the version probe.
    (site / "probe_test.pth").write_text(
        f"import pathlib; pathlib.Path({str(marker)!r}).write_text('initialized')\n"
    )
    subprocess.run([str(python), "-I", "-c", "pass"], check=True, timeout=15,
                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    assert marker.exists()
    marker.unlink()
    if sys.version_info[:2] in ((3, 10), (3, 11)):
        selector.check_python(python)
    else:
        with pytest.raises(RuntimeError, match="requires 3.10/3.11"):
            selector.check_python(python)
    assert not marker.exists()
