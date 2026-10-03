import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from envs.swebench_apptainer import ApptainerSandbox, checked_image, image_record
from scripts import run_swe_arm_subset as runner
from scripts.grade_swe_arm_pilot import checked_eval_script


def dataset_fixture(tmp_path):
    rows = [dict(instance_id=f"django__django-{i}", repo="django/django", base_commit="a" * 40,
                 problem_statement="Fix a bug") for i in range(300)]
    inventory = {"dataset_revision": "b" * 40, "instances": [
        dict(instance_id=r["instance_id"], status="available" if i < 2 else "registry_error",
             architecture="arm64", os="linux", manifest_digest="sha256:" + "c" * 64)
        for i, r in enumerate(rows)]}
    data = tmp_path / "data"
    (data / "public").mkdir(parents=True)
    (data / "grading").mkdir()
    runner.save(data / "public/instances.json", rows)
    runner.save(data / "grading/instances.json", [dict(row, patch="gold", test_patch="tests") for row in rows])
    manifest = {"dataset": "princeton-nlp/SWE-bench_Lite", "count": 300, "split": "test",
                "revision": "b" * 40, "public_sha256": runner.file_hash(data / "public/instances.json"),
                "grading_sha256": runner.file_hash(data / "grading/instances.json")}
    runner.save(data / "manifest.json", manifest)
    inventory_path = tmp_path / "inventory.json"
    runner.save(inventory_path, inventory)
    return rows, manifest, inventory, data, inventory_path


def test_inventory_selection_preserves_exclusions_and_pins_dataset(tmp_path):
    rows, manifest, inventory, _, _ = dataset_fixture(tmp_path)
    ids, records = runner.selection(rows, manifest, inventory, -1)
    assert len(ids) == 2 and records[ids[0]]["oci_digest"] == "c" * 64
    with pytest.raises(ValueError, match="pinned Lite"):
        runner.selection(rows, dict(manifest, revision="d" * 40), inventory, -1)
    inventory["instances"][2]["instance_id"] = ids[0]
    with pytest.raises(ValueError, match="300 Lite"):
        runner.selection(rows, manifest, inventory, -1)


def test_generalized_sandbox_checks_task_identity_and_uses_task_base(tmp_path, monkeypatch):
    rows, manifest, inventory, _, _ = dataset_fixture(tmp_path)
    _, records = runner.selection(rows, manifest, inventory, -1)
    runner.save(tmp_path / "arm-images.json", records)
    task = rows[0]
    name, _ = image_record(tmp_path, task)
    (tmp_path / "images").mkdir()
    image = tmp_path / "images" / name
    image.write_bytes(b"fixture")
    Path(str(image) + ".sha256").write_text(hashlib.sha256(b"fixture").hexdigest() + "  " + name)
    assert checked_image(tmp_path, task)[0] == image
    with pytest.raises(ValueError, match="identity mismatch"):
        checked_image(tmp_path, dict(task, base_commit="f" * 40))
    sandbox = ApptainerSandbox(task, root=tmp_path)
    calls = []
    monkeypatch.setattr(sandbox, "_exec", lambda command, **kw: (calls.append(command) or 0, "patch"))
    assert sandbox.patch() == "patch"
    assert "a" * 40 in calls[0]


@pytest.mark.skipif(not shutil.which("bash"), reason="Bash required")
@pytest.mark.parametrize("setup_exit", [0, 1])
def test_generic_grading_setup_gate_before_tests(tmp_path, setup_exit):
    source = "set -uxo pipefail\nsetup_command\n: '>>>>> Start Test Output'\nfalse\necho TESTS_DONE\n"
    script = checked_eval_script(source, "django/django")
    assert "import os,pathlib,sys,django" in script and "is_relative_to" not in script
    target = tmp_path / "eval.sh"
    target.write_text(f"setup_command() {{ return {setup_exit}; }}\npython() {{ return 0; }}\n" + script, newline="\n")
    proc = subprocess.run([shutil.which("bash"), target.as_posix()], capture_output=True,
                          creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    assert (b"TESTS_DONE" in proc.stdout) == (setup_exit == 0)


@pytest.mark.parametrize("calibration_fails", [False, True])
@pytest.mark.parametrize("method", ["contextgraph", "foldagent"])
def test_batch_calibrates_before_generation_and_resumes_without_repeating(tmp_path, monkeypatch, calibration_fails, method):
    rows, _, _, data, inventory = dataset_fixture(tmp_path)
    output = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", ["runner", "--data-dir", str(data), "--output", str(output),
        "--inventory", str(inventory), "--agent-python", "agent-python", "--model-path", str(tmp_path),
        "--endpoint", "http://test", "--method", method])
    monkeypatch.setattr(runner, "require_harness", lambda: None)
    monkeypatch.setitem(sys.modules, "fcntl", SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=lambda *a: None))
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **k: "commit")
    monkeypatch.setattr(runner.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(
        b'{"data":[{"id":"Qwen/Qwen3.5-9B","max_model_len":65536}]}'))
    events = []

    def build(task, root, attempt):
        events.append((task["instance_id"], "build"))
        return Path("image"), "sha"

    def evaluate(task, patch, folder, root, timeout, **kwargs):
        events.append((task["instance_id"], folder.name))
        assert kwargs["runtime_template"] == folder.parent / "build-environment/runtime"
        assert kwargs["runtime_template"].is_dir()
        folder.mkdir()
        resolved = bool(patch) and not (calibration_fails and task["instance_id"] == rows[0]["instance_id"])
        return {task["instance_id"]: {"resolved": resolved}}

    def prepare_build(task, folder, root, timeout):
        events.append((task["instance_id"], "build_dependencies"))
        template = folder / "runtime"
        template.mkdir(parents=True)
        runner.save(folder / "environment.json", {"prepared": True})
        return template

    def generate(argv, log, **kwargs):
        assert argv[argv.index("--method") + 1] == method
        instance = argv[argv.index("--instance-ids") + 1]
        events.append((instance, "generation"))
        gen = Path(argv[argv.index("--output") + 1])
        gen.mkdir()
        runner.save(gen / "manifest.json", {"instance_ids": [instance], "dataset": runner.read_json(data / "manifest.json")})
        (gen / "predictions.jsonl").write_text(json.dumps({"instance_id": instance, "model_patch": "predicted",
                                                        "model_name_or_path": "Qwen"}) + "\n")
        (gen / "results.jsonl").write_text(json.dumps({"instance_id": instance, "status": "generated",
                                                     "image": {"sif_sha256": "sha"}}) + "\n")
        runner.save(gen / "generation_summary.json", {"predictions_sha256": runner.file_hash(gen / "predictions.jsonl")})

    monkeypatch.setattr(runner, "build_image", build)
    monkeypatch.setattr(runner, "prepare_build_environment", prepare_build)
    monkeypatch.setattr(runner, "evaluate", evaluate)
    monkeypatch.setattr(runner, "run_command", generate)
    monkeypatch.setattr(runner, "checked_image", lambda *a: (Path("image"), "sha"))
    assert runner.main() == (2 if calibration_fails else 0)
    result = runner.read_json(output / "summary.json")
    assert runner.read_json(output / "manifest.json")["method"] == method
    assert result["selected"] == result["completed"] == 2
    assert result["graded"] == (1 if calibration_fails else 2)
    assert result["official_x86_result"] is False
    first = [stage for instance, stage in events if instance == rows[0]["instance_id"]]
    assert first == (["build", "build_dependencies", "baseline", "reference"] if calibration_fails else
                     ["build", "build_dependencies", "baseline", "reference", "generation", "grading"])
    assert not list(output.glob("instances/*/attempt-*/build-environment/runtime"))
    before = list(events)
    runner.main()
    assert events == before
    sys.argv[sys.argv.index("--method") + 1] = "react"
    with pytest.raises(ValueError, match="Resume protocol"):
        runner.main()


@pytest.mark.skipif(not shutil.which("bash"), reason="Bash required")
def test_subset_launcher_syntax():
    script = Path(__file__).resolve().parents[1] / "scripts/run_swe_lite_arm_subset_idev.sh"
    subprocess.run([shutil.which("bash"), "-n", script.as_posix()], check=True,
                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)


def test_real_inventory_shards_cover_252_without_overlap(tmp_path):
    inventory = runner.read_json(runner.REPO / "configs/swe_lite_arm_images.json")
    rows = [dict(instance_id=e["instance_id"], repo="unused", base_commit="a" * 40) for e in inventory["instances"]]
    dataset = dict(dataset="princeton-nlp/SWE-bench_Lite", revision=inventory["dataset_revision"])
    first, _ = runner.selection(rows, dataset, inventory, -1, 0, 2)
    second, _ = runner.selection(rows, dataset, inventory, -1, 1, 2)
    all_ids, _ = runner.selection(rows, dataset, inventory, -1)
    assert len(first) == len(second) == 126
    assert not set(first).intersection(second)
    assert sorted(first + second) == all_ids
    limited = [runner.selection(rows, dataset, inventory, 5, i, 2)[0] for i in range(2)]
    assert sorted(limited[0] + limited[1]) == all_ids[:5]
    with pytest.raises(ValueError, match="shard"):
        runner.selection(rows, dataset, inventory, -1, 2, 2)


@pytest.mark.skipif(not shutil.which("bash"), reason="Bash required")
def test_four_node_launcher_syntax():
    subprocess.run([shutil.which("bash"), "-n", (runner.REPO / "scripts/run_swe_lite_arm_4node_idev.sh").as_posix()],
                   check=True, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)


def test_offline_editable_build_reuses_image_dependencies(tmp_path):
    """Exercise real pip: default isolation fails, generated policy builds locally.

    A self-contained PEP 517 backend avoids depending on host setuptools/wheel.
    The unavailable build dependency stands in for Astropy's offline bootstrap.
    """
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        '[build-system]\nrequires = ["swe-offline-missing-build-dependency==0.0.0"]\n'
        'build-backend = "fixture_backend"\nbackend-path = ["."]\n')
    (project / "fixture_backend.py").write_text(
        'import pathlib, zipfile\n'
        'def build_editable(wheel_directory, config_settings=None, metadata_directory=None):\n'
        '    name = "swe_offline_fixture-1.0-py3-none-any.whl"\n'
        '    with zipfile.ZipFile(pathlib.Path(wheel_directory) / name, "w") as wheel:\n'
        '        wheel.writestr("swe_offline_fixture.py", "VALUE = 42\\n")\n'
        '        wheel.writestr("swe_offline_fixture-1.0.dist-info/METADATA", '
        '"Metadata-Version: 2.1\\nName: swe-offline-fixture\\nVersion: 1.0\\n")\n'
        '        wheel.writestr("swe_offline_fixture-1.0.dist-info/WHEEL", '
        '"Wheel-Version: 1.0\\nGenerator: fixture\\nRoot-Is-Purelib: true\\nTag: py3-none-any\\n")\n'
        '        wheel.writestr("swe_offline_fixture-1.0.dist-info/RECORD", "")\n'
        '    return name\n')
    source = "set -uxo pipefail\npython -m pip install -e '.[test]' --verbose\n: '>>>>> Start Test Output'\ntrue\n"
    rendered = checked_eval_script(source, "astropy/astropy")
    assert "python -m pip install -e '.[test]' --verbose" in rendered
    env = {k: v for k, v in os.environ.items() if not k.startswith("PIP_")}
    env.update(PIP_CONFIG_FILE=os.devnull, PIP_NO_INDEX="1", PIP_DISABLE_PIP_VERSION_CHECK="1", PIP_RETRIES="0")
    argv = [sys.executable, "-m", "pip", "install", "--no-cache-dir", "--target", str(tmp_path / "installed"), "-e", ".[test]"]
    options = dict(cwd=project, env=env, capture_output=True, timeout=60,
                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    failed = subprocess.run(argv, **options)
    assert failed.returncode != 0
    assert b"swe-offline-missing-build-dependency" in failed.stdout + failed.stderr
    for line in rendered.splitlines():
        if line.startswith("export PIP_"):
            env.update(item.split("=", 1) for item in line.removeprefix("export ").split())
    success = subprocess.run(argv, **options)
    assert success.returncode == 0, (success.stdout + success.stderr).decode(errors="replace")
    assert (tmp_path / "installed/swe_offline_fixture.py").read_text() == "VALUE = 42\n"
