"""Single-instance ARM compatibility grading with pinned upstream test logic."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from envs.swebench_apptainer import ApptainerSandbox, INSTANCE, BASE
from scripts.eval_swebench_verified import (DATASETS, HARNESS_COMMIT, file_hash, load_public,
    read_json, read_jsonl, validate_predictions, write_json)


def load_task(data_dir, benchmark=None):
    _, manifest = load_public(data_dir)
    if benchmark and manifest["dataset"] != DATASETS[benchmark][0]:
        raise ValueError("Prepared dataset does not match requested benchmark")
    gold = data_dir / "grading/instances.json"
    if file_hash(gold) != manifest["grading_sha256"]:
        raise ValueError("Grading dataset checksum mismatch")
    rows = [r for r in read_json(gold) if r["instance_id"] == INSTANCE]
    if len(rows) != 1 or rows[0]["base_commit"] != BASE:
        raise ValueError("Missing pinned pilot task")
    return rows[0], manifest


def require_harness():
    dist = importlib.metadata.distribution("swebench")
    source = json.loads(dist.read_text("direct_url.json") or "{}")
    if source.get("vcs_info", {}).get("commit_id") != HARNESS_COMMIT:
        raise RuntimeError("Install requirements_swebench_eval.txt in the grading venv")


def checked_eval_script(script):
    """Fail on setup errors, but retain upstream handling of failed tests."""
    header = "set -uxo pipefail\n"
    install = "python -m pip install -e .\n"
    marker = ": '>>>>> Start Test Output'\n"
    if any(script.count(part) != 1 for part in (header, install, marker)):
        raise RuntimeError("Pinned SymPy evaluation script changed; review setup boundaries")
    check = ('python -c "import pathlib,sys,sympy; '
             'p=pathlib.Path(sympy.__file__).resolve(); '
             "assert p.is_relative_to(pathlib.Path('/testbed')), str(p); "
             "assert sys.prefix == '/opt/miniconda3/envs/testbed', sys.prefix; "
             "print('SWE_ARM_IMPORT_OK', p)" + '"\n')
    # --no-home leaves the host HOME path unavailable. Use the container's private
    # /tmp for Git/Conda/pip user configuration, outside the submitted repository.
    setup = ('set -euxo pipefail\n'
             'HOME=$(mktemp -d /tmp/swe-grade-home-XXXXXX)\n'
             'export HOME\n'
             'export XDG_CONFIG_HOME="$HOME/.config"\n'
             'mkdir -p "$XDG_CONFIG_HOME"\n')
    return (script.replace(header, setup)
            .replace(install, install + check)
            .replace(marker, "set +e\n" + marker))


def evaluate(task, patch, folder, root, timeout):
    from swebench.harness.test_spec.test_spec import make_test_spec
    from swebench.harness.grading import get_eval_report, get_logs_eval
    folder.mkdir(parents=True, exist_ok=False)
    inputs = folder / "inputs"
    inputs.mkdir()
    spec = make_test_spec(task)
    (folder / "upstream_eval.sh").write_text(spec.eval_script, encoding="utf8", newline="\n")
    (inputs / "eval.sh").write_text(checked_eval_script(spec.eval_script), encoding="utf8", newline="\n")
    (inputs / "model.patch").write_text(patch, encoding="utf8", newline="\n")
    prediction = {"instance_id": INSTANCE, "model_name_or_path": "arm-pilot", "model_patch": patch}
    sandbox = ApptainerSandbox(task, root=root)
    try:
        sandbox.start()
        sandbox.prepare_grading_environment()
        write_json(folder / "image.json", sandbox.provenance)
        if patch.strip():
            status, text = sandbox._exec("echo SWE_ARM_APPLY_STARTED; git apply --verbose /grading-input/model.patch", inputs=inputs)
            (folder / "apply.log").write_text(text, encoding="utf8")
            if status in (124, 137) or "[output truncated]" in text:
                raise RuntimeError("Patch application timed out or output exceeded limit")
            if "SWE_ARM_APPLY_STARTED" not in text:
                raise RuntimeError("Container did not reach patch application")
            if status:
                report = {INSTANCE: {"resolved": False, "patch_successfully_applied": False}}
                write_json(folder / "report.json", report)
                return report
        status, text = sandbox._exec("/bin/bash /grading-input/eval.sh", inputs=inputs,
                                     timeout=timeout, limit=32 * 1024 * 1024)
        log = folder / "test_output.txt"
        log.write_text(text, encoding="utf8")
        if status in (124, 137) or text.endswith("\n[output truncated]"):
            raise RuntimeError("Grading timeout or truncated output; not a model failure")
        if status or not any(line.startswith("SWE_ARM_IMPORT_OK /testbed/") for line in text.splitlines()):
            raise RuntimeError(f"Grading setup/import verification failed (exit {status}); "
                               f"log={log}\nLast output:\n{text[-4000:]}")
        statuses, found = get_logs_eval(spec, str(log))
        if not found or not set(statuses).intersection(spec.FAIL_TO_PASS + spec.PASS_TO_PASS):
            raise RuntimeError("Missing parseable task test results; inspect test_output.txt")
        report = get_eval_report(spec, prediction, str(log), include_tests_status=True)
        write_json(folder / "report.json", report)
        return report
    finally:
        sandbox.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--dataset", choices=DATASETS)
    p.add_argument("--apptainer-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--calibrate", action="store_true")
    p.add_argument("--test-timeout", type=int, default=1800)
    args = p.parse_args()
    require_harness()
    task, dataset = load_task(args.data_dir, args.dataset)
    if args.calibrate:
        args.output.mkdir(parents=True, exist_ok=False)
        baseline = evaluate(task, "", args.output / "baseline", args.apptainer_root, args.test_timeout)
        gold = evaluate(task, task["patch"], args.output / "reference", args.apptainer_root, args.test_timeout)
        if baseline[INSTANCE]["resolved"] or not gold[INSTANCE]["resolved"]:
            raise RuntimeError("ARM calibration failed: require baseline unresolved and reference resolved")
        write_json(args.output / "calibration.json", {"instance_id": INSTANCE, "calibration_passed": True,
            "dataset": dataset, "harness_commit": HARNESS_COMMIT, "official_x86_result": False})
        print("SWE_ARM_CALIBRATION_PASSED", args.output)
        return
    manifest, predictions = validate_predictions(args.output)
    if manifest.get("backend") != "apptainer" or manifest["instance_ids"] != [INSTANCE] or manifest["dataset"] != dataset:
        raise ValueError("Expected matching single-instance ARM generation run")
    grading = args.output / "grading-arm"
    prediction = predictions[0]
    # Verify the generation SIF identity before grading in a fresh workspace.
    from envs.swebench_apptainer import checked_image
    _, sha = checked_image(args.apptainer_root, task)
    if read_jsonl(args.output / "results.jsonl")[0]["image"]["sif_sha256"] != sha:
        raise ValueError("SIF image changed after generation")
    if prediction["model_patch"].strip():
        report = evaluate(task, prediction["model_patch"], grading, args.apptainer_root, args.test_timeout)
        resolved = report[INSTANCE]["resolved"]
    else:
        grading.mkdir(parents=True, exist_ok=False)
        resolved = False
    summary = {"instance_id": INSTANCE, "count": 1, "resolved": int(resolved),
        "empty_patch": not prediction["model_patch"].strip(), "grading_complete": True,
        "runtime": "apptainer-arm-pilot", "official_x86_result": False, "harness_commit": HARNESS_COMMIT,
        "predictions_sha256": file_hash(args.output / "predictions.jsonl"), "sif_sha256": sha,
        "dataset": dataset}
    write_json(grading / "summary.json", summary)
    print("SWE_ARM_PILOT_COMPLETE", json.dumps(summary))


if __name__ == "__main__":
    main()
