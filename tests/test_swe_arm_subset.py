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
def test_batch_calibrates_before_generation_and_resumes_without_repeating(tmp_path, monkeypatch, calibration_fails):
    rows, _, _, data, inventory = dataset_fixture(tmp_path)
    output = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", ["runner", "--data-dir", str(data), "--output", str(output),
        "--inventory", str(inventory), "--agent-python", "agent-python", "--model-path", str(tmp_path),
        "--endpoint", "http://test"])
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
        folder.mkdir()
        resolved = bool(patch) and not (calibration_fails and task["instance_id"] == rows[0]["instance_id"])
        return {task["instance_id"]: {"resolved": resolved}}

    def generate(argv, log, **kwargs):
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
    monkeypatch.setattr(runner, "evaluate", evaluate)
    monkeypatch.setattr(runner, "run_command", generate)
    monkeypatch.setattr(runner, "checked_image", lambda *a: (Path("image"), "sha"))
    assert runner.main() == (2 if calibration_fails else 0)
    result = runner.read_json(output / "summary.json")
    assert result["selected"] == result["completed"] == 2
    assert result["graded"] == (1 if calibration_fails else 2)
    assert result["official_x86_result"] is False
    first = [stage for instance, stage in events if instance == rows[0]["instance_id"]]
    assert first == (["build", "baseline", "reference"] if calibration_fails else
                     ["build", "baseline", "reference", "generation", "grading"])
    before = list(events)
    runner.main()
    assert events == before


@pytest.mark.skipif(not shutil.which("bash"), reason="Bash required")
def test_subset_launcher_syntax():
    script = Path(__file__).resolve().parents[1] / "scripts/run_swe_lite_arm_subset_idev.sh"
    subprocess.run([shutil.which("bash"), "-n", script.as_posix()], check=True,
                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
