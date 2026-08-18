#!/usr/bin/env python3
"""Cache or verify the D3-Gym task images referenced by verl parquets."""

from __future__ import annotations

import argparse
import ast
import json
import os
import platform
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


def _extra_info(value) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(value)
            except (SyntaxError, TypeError, ValueError):
                continue
            if isinstance(parsed, dict):
                return parsed
    return {}


def collect_task_ids(parquets: list[str], explicit: str | None = None) -> list[str]:
    task_ids: list[str] = []
    seen: set[str] = set()

    def add(value) -> None:
        task_id = str(value).strip() if value is not None else ""
        if task_id and task_id not in seen:
            seen.add(task_id)
            task_ids.append(task_id)

    for parquet in parquets:
        frame = pd.read_parquet(parquet, columns=["extra_info"])
        for value in frame["extra_info"]:
            add(_extra_info(value).get("task_id"))
    if explicit:
        for value in explicit.split(","):
            add(value)
    return task_ids


def image_path(image_dir: str, task_id: str) -> Path:
    return Path(image_dir, f"{task_id}.sif")


def canonical_arch(value: str) -> str:
    aliases = {
        "x86_64": "amd64",
        "x64": "amd64",
        "aarch64": "arm64",
        "arm64v8": "arm64",
    }
    normalized = str(value or "").strip().lower()
    return aliases.get(normalized, normalized)


def inspect_image_arch(runtime: str, image: Path) -> tuple[str | None, str]:
    try:
        proc = subprocess.run(
            [runtime, "inspect", "--json", str(image)],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return None, "image metadata inspection timed out"
    if proc.returncode != 0:
        return None, (proc.stderr or proc.stdout or "image metadata inspection failed").strip()[-1000:]
    try:
        payload = json.loads(proc.stdout)
        labels = payload["data"]["attributes"]["labels"]
        arch = labels.get("org.label-schema.build-arch")
    except (KeyError, TypeError, ValueError):
        return None, "org.label-schema.build-arch is missing from image metadata"
    return canonical_arch(arch), ""


def pull_image(runtime: str, image_dir: str, image_template: str, task_id: str) -> tuple[str, bool, str]:
    destination = image_path(image_dir, task_id)
    if destination.is_file() and destination.stat().st_size > 0:
        return task_id, True, "cached"
    partial = destination.with_name(f".{task_id}.{os.getpid()}.partial.sif")
    partial.unlink(missing_ok=True)
    source = image_template.format(task_id=task_id)
    if "://" not in source:
        source = "docker://" + source
    proc = subprocess.run(
        [runtime, "pull", "--force", str(partial), source],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        partial.unlink(missing_ok=True)
        return task_id, False, (proc.stderr or proc.stdout or "pull failed").strip()[-1000:]
    if not partial.is_file() or partial.stat().st_size == 0:
        partial.unlink(missing_ok=True)
        return task_id, False, "runtime reported success but did not create a non-empty SIF"
    os.replace(partial, destination)
    return task_id, True, "downloaded"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", nargs="*", default=[])
    parser.add_argument("--task-ids", help="Additional comma-separated task IDs")
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--runtime", choices=("apptainer", "singularity"), default="apptainer")
    parser.add_argument("--image-template", default="hananemoussa/d3-gym:{task_id}")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--check-arch", action="store_true")
    args = parser.parse_args()

    task_ids = collect_task_ids(args.parquet, args.task_ids)
    if args.limit is not None:
        task_ids = task_ids[: args.limit]
    if not task_ids:
        raise SystemExit("ERROR: no D3-Gym task IDs were found")
    os.makedirs(args.image_dir, exist_ok=True)
    missing = [
        task_id for task_id in task_ids
        if not image_path(args.image_dir, task_id).is_file()
        or image_path(args.image_dir, task_id).stat().st_size == 0
    ]
    if args.check_only:
        if missing:
            print(f"ERROR: {len(missing)}/{len(task_ids)} D3-Gym images are missing from {args.image_dir}")
            print("missing: " + " ".join(missing[:20]) + (" ..." if len(missing) > 20 else ""))
            raise SystemExit(1)
        print(f"[OK] {len(task_ids)} D3-Gym images present in {args.image_dir}")
        if args.check_arch:
            if shutil.which(args.runtime) is None:
                raise SystemExit(f"ERROR: runtime executable not found: {args.runtime}")
            host_arch = canonical_arch(platform.machine())
            failures = []
            for task_id in task_ids:
                arch, detail = inspect_image_arch(args.runtime, image_path(args.image_dir, task_id))
                if arch is None:
                    failures.append(f"{task_id}: unknown ({detail})")
                elif arch != host_arch:
                    failures.append(f"{task_id}: image={arch} host={host_arch}")
            if failures:
                print(
                    "ERROR: D3-Gym image architecture is incompatible with this host; "
                    "native matching images or x86_64 compute nodes are required."
                )
                print("architecture failures: " + "; ".join(failures[:20]))
                raise SystemExit(2)
            print(f"[OK] D3-Gym image architecture matches host: {host_arch}")
        return

    if shutil.which(args.runtime) is None:
        raise SystemExit(f"ERROR: runtime executable not found: {args.runtime}")
    failures = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(pull_image, args.runtime, args.image_dir, args.image_template, task_id): task_id
            for task_id in missing
        }
        for future in as_completed(futures):
            task_id, ok, detail = future.result()
            print(f"[{'OK' if ok else 'FAIL'}] {task_id}: {detail}", flush=True)
            if not ok:
                failures.append(task_id)
    if failures:
        raise SystemExit(f"ERROR: failed to cache {len(failures)} images")
    print(f"[OK] D3-Gym image cache complete: {len(task_ids)} tasks in {args.image_dir}")


if __name__ == "__main__":
    main()
