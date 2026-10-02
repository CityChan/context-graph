"""Sequential, resumable ARM compatibility subset; never an official x86 score."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import traceback
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from envs.swebench_apptainer import checked_image, image_record
from scripts.eval_swebench_verified import file_hash, load_public, read_json, read_jsonl, validate_predictions
from scripts.grade_swe_arm_pilot import evaluate, prepare_build_environment, require_harness

REPO = Path(__file__).resolve().parents[1]


def save(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf8")
    temp.replace(path)


def selection(rows, dataset, inventory, limit):
    if dataset["dataset"] != "princeton-nlp/SWE-bench_Lite" or dataset["revision"] != inventory["dataset_revision"]:
        raise ValueError("ARM inventory requires the pinned Lite dataset revision")
    entries = inventory["instances"]
    if len(entries) != 300 or {r["instance_id"] for r in entries} != {r["instance_id"] for r in rows}:
        raise ValueError("Inventory must account for all 300 Lite tasks exactly once")
    records = {}
    public = {r["instance_id"]: r for r in rows}
    for entry in entries:
        if entry["status"] != "available":
            continue
        if entry.get("architecture") != "arm64" or entry.get("os") != "linux":
            raise ValueError("Inventory platform mismatch")
        instance = entry["instance_id"]
        digest = entry["manifest_digest"]
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
            raise ValueError("Missing manifest digest")
        records[instance] = {"repo": public[instance]["repo"], "base_commit": public[instance]["base_commit"],
                             "oci_digest": digest.removeprefix("sha256:")}
    ids = sorted(records)
    if limit != -1:
        if not 1 <= limit <= len(ids):
            raise ValueError("--limit must be -1 or within the available subset size")
        ids = ids[:limit]
    return ids, {key: records[key] for key in ids}


def run_command(argv, log, *, timeout, env=None):
    """Terminate the entire child process group on timeout or interrupted allocation."""
    with Path(log).open("wb") as handle:
        proc = subprocess.Popen(argv, cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            status = proc.wait(timeout=timeout)
        except BaseException:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
            raise
    if status:
        raise RuntimeError(f"Child exited {status}; inspect {log}")


def build_image(task, root, attempt):
    name, digest = image_record(root, task)
    image = root / "images" / name
    if not image.exists():
        # Download/build one task at a time; --disable-cache avoids retaining OCI layers.
        temporary = attempt / "image.sif"
        env = dict(os.environ, APPTAINER_CACHEDIR=str(root / "apptainer-cache"),
                   APPTAINER_TMPDIR=str(attempt / "tmp"), GOMAXPROCS="1")
        for key in ("LD_PRELOAD", "APPTAINER_BIND", "APPTAINER_BINDPATH", "SINGULARITY_BIND", "SINGULARITY_BINDPATH"):
            env.pop(key, None)
        (attempt / "tmp").mkdir(exist_ok=True)
        uri = f'docker://ghcr.io/epoch-research/swe-bench.eval.arm64.{task["instance_id"]}@sha256:{digest}'
        try:
            run_command(["apptainer", "build", "--disable-cache", "--mksquashfs-args", "-processors 1 -mem 256M",
                         str(temporary), uri], attempt / "image-build.log", timeout=3600, env=env)
            if not temporary.is_file() or not temporary.stat().st_size:
                raise RuntimeError("Apptainer did not produce an image")
            digestor = hashlib.sha256()
            with temporary.open("rb") as handle:
                for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digestor.update(block)
            sha = digestor.hexdigest()
            temporary.replace(image)
            Path(str(image) + ".sha256").write_text(f"{sha}  {name}\n")
        finally:
            temporary.unlink(missing_ok=True)
            shutil.rmtree(attempt / "tmp")
    return checked_image(root, task)


def summary(run, ids):
    results = [read_json(run / "instances" / key / "result.json") for key in ids
               if (run / "instances" / key / "result.json").exists()]
    counts = Counter(r["status"] for r in results)
    value = {"runtime": "apptainer-arm-subset", "official_x86_result": False,
             "selected": len(ids), "completed": len(results), "pending": len(ids) - len(results),
             "status_counts": dict(counts), "resolved": sum(r.get("resolved", False) for r in results),
             "graded": counts["graded"], "scope": "image-available subset; infrastructure errors reported separately"}
    save(run / "summary.json", value)
    print("SWE_ARM_SUBSET_PROGRESS", json.dumps(value), flush=True)
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--agent-python", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--endpoint", required=True)
    p.add_argument("--inventory", type=Path, default=REPO / "configs/swe_lite_arm_images.json")
    p.add_argument("--context-length", type=int, default=65536)
    p.add_argument("--limit", type=int, default=-1)
    p.add_argument("--test-timeout", type=int, default=1800)
    p.add_argument("--retry-errors", action="store_true")
    args = p.parse_args()
    if args.context_length not in (32768, 65536):
        p.error("Expected 32768 or 65536 context tokens")
    require_harness()
    rows, dataset = load_public(args.data_dir)
    inventory = read_json(args.inventory)
    ids, records = selection(rows, dataset, inventory, args.limit)
    gold_path = args.data_dir / "grading/instances.json"
    if file_hash(gold_path) != dataset["grading_sha256"]:
        raise ValueError("Grading dataset checksum mismatch")
    gold = {r["instance_id"]: r for r in read_json(gold_path)}
    public = {r["instance_id"]: r for r in rows}
    for key in ids:
        if any(gold[key][field] != public[key][field] for field in public[key]):
            raise ValueError("Public/grading task mismatch")
    with urllib.request.urlopen(args.endpoint.rstrip("/") + "/v1/models", timeout=15) as response:
        models = json.load(response)["data"]
    if not any(m["id"] == "Qwen/Qwen3.5-9B" and m.get("max_model_len", 0) >= args.context_length for m in models):
        raise ValueError("Expected live Qwen/Qwen3.5-9B server with requested context length")
    protocol = {"dataset": dataset, "instance_ids": ids, "inventory_sha256": file_hash(args.inventory),
                "context_length": args.context_length, "method": "contextgraph", "seed": 42,
                "test_timeout": args.test_timeout, "model_path": str(Path(args.model_path).resolve()),
                "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                "official_x86_result": False}
    run = args.output.resolve()
    run.mkdir(parents=True, exist_ok=True)
    # One owner per run; concurrent resume must not delete another owner's image.
    import fcntl
    with (run / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (run / "manifest.json").exists():
            if read_json(run / "manifest.json") != protocol:
                raise ValueError("Resume protocol/commit mismatch; use a fresh output directory")
        else:
            save(run / "manifest.json", protocol)
            save(run / "image-inventory.json", inventory)
        root = run / "runtime"
        (root / "images").mkdir(parents=True, exist_ok=True)
        save(root / "arm-images.json", records)
        summary(run, ids)
        for index, instance in enumerate(ids, 1):
            folder = run / "instances" / instance
            result_file = folder / "result.json"
            if result_file.exists() and (not args.retry_errors or read_json(result_file)["status"] == "graded"):
                continue
            folder.mkdir(parents=True, exist_ok=True)
            result_file.unlink(missing_ok=True)
            attempt = Path(tempfile.mkdtemp(prefix="attempt-", dir=folder))
            task = gold[instance]
            stage = "image"
            print(f"SWE_ARM_SUBSET_TASK {index}/{len(ids)} {instance} artifacts={attempt}", flush=True)
            result = {"instance_id": instance, "attempt": str(attempt), "status": "infrastructure_error"}
            runtime_template = None
            try:
                _, sha = build_image(task, root, attempt)
                result["sif_sha256"] = sha
                stage = "build_dependencies"
                print(f"SWE_ARM_SUBSET_STAGE {instance} build_dependencies", flush=True)
                runtime_template = prepare_build_environment(task, attempt / "build-environment", root, args.test_timeout)
                result["build_environment_record_sha256"] = file_hash(runtime_template.parent / "environment.json")
                stage = "calibration"
                print(f"SWE_ARM_SUBSET_STAGE {instance} calibration", flush=True)
                baseline = evaluate(task, "", attempt / "baseline", root, args.test_timeout, subset=True, runtime_template=runtime_template)
                reference = evaluate(task, task["patch"], attempt / "reference", root, args.test_timeout, subset=True, runtime_template=runtime_template)
                if baseline[instance]["resolved"] or not reference[instance]["resolved"]:
                    raise RuntimeError("Require baseline unresolved and reference resolved")
                save(attempt / "calibration.json", {"passed": True, "sif_sha256": sha})
                stage = "generation"
                print(f"SWE_ARM_SUBSET_STAGE {instance} generation", flush=True)
                generation = attempt / "generation"
                env = dict(os.environ)
                # Only the agent needs the Vista Torch TLS preload; never load it into grading.
                preload = env.pop("SWE_AGENT_LD_PRELOAD", "")
                if preload:
                    env["LD_PRELOAD"] = preload
                run_command([args.agent_python, "-u", "scripts/eval_swebench_verified.py", "generate",
                             "--dataset", "lite", "--backend", "apptainer", "--apptainer-root", str(root),
                             "--data-dir", str(args.data_dir.resolve()), "--output", str(generation),
                             "--method", "contextgraph", "--model-path", args.model_path, "--endpoint", args.endpoint,
                             "--context-length", str(args.context_length), "--instance-ids", instance,
                             "--samples", "1", "--workers", "1", "--seed", "42"],
                            attempt / "generation.log", timeout=7200, env=env)
                manifest, predictions = validate_predictions(generation)
                if manifest["instance_ids"] != [instance] or manifest["dataset"] != dataset:
                    raise ValueError("Unexpected generation task/dataset")
                if read_jsonl(generation / "results.jsonl")[0]["image"]["sif_sha256"] != sha:
                    raise ValueError("Generation image mismatch")
                if checked_image(root, task)[1] != sha:
                    raise ValueError("Image changed before grading")
                stage = "grading"
                print(f"SWE_ARM_SUBSET_STAGE {instance} grading", flush=True)
                patch = predictions[0]["model_patch"]
                report = evaluate(task, patch, attempt / "grading", root, args.test_timeout, subset=True, runtime_template=runtime_template)
                result.update(status="graded", resolved=bool(report[instance]["resolved"]),
                              empty_patch=not patch.strip(), predictions_sha256=file_hash(generation / "predictions.jsonl"))
            except Exception as exc:
                result.update(stage=stage, error=str(exc))
                (attempt / "error.txt").write_text(traceback.format_exc(), encoding="utf8")
                print(f"SWE_ARM_SUBSET_ERROR {instance} stage={stage}: {exc}", flush=True)
            except BaseException:
                save(attempt / "interrupted.json", {"stage": stage, "instance_id": instance})
                raise
            finally:
                # Bound disk use to one image. Preserve every log, patch and digest.
                if runtime_template is not None:
                    shutil.rmtree(runtime_template)
                name, _ = image_record(root, task)
                image = root / "images" / name
                image.unlink(missing_ok=True)
                Path(str(image) + ".sha256").unlink(missing_ok=True)
            save(result_file, result)
            summary(run, ids)
        value = summary(run, ids)
        print("SWE_ARM_SUBSET_COMPLETE", run, flush=True)
        return 0 if value["graded"] == len(ids) else 2


if __name__ == "__main__":
    def stop(signum, frame):
        raise KeyboardInterrupt(f"Interrupted by signal {signum}; completed tasks can be resumed")
    signal.signal(signal.SIGTERM, stop)
    sys.exit(main())
