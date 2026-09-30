"""Prepare pinned data and check the Vista agent environment; no evaluation."""
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.eval_swebench_verified import file_hash, load_public, prepare

REVISION = "c104f840cc67f8b6eec6f759ebc8b2693d585d4a"


def check_data(root):
    rows, manifest = load_public(root)
    if manifest["revision"] != REVISION:
        raise ValueError("Existing dataset revision differs from the pinned Vista probe")
    if file_hash(root / "grading/instances.json") != manifest["grading_sha256"]:
        raise ValueError("Grading dataset hash mismatch")
    probe = next(row for row in rows if row["instance_id"] == "sympy__sympy-20590")
    if probe["base_commit"] != "cffd4e0f86fefd4802349a9f9b19ed70934ea354":
        raise ValueError("Dataset base commit differs from the pinned ARM image probe")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    # Run in the isolated preparation venv: newer Transformers ignores USE_TORCH.
    # These flags also disable backend discovery in older Transformers/datasets.
    for name in ("USE_TORCH", "USE_TF", "USE_FLAX"):
        os.environ[name] = "0"
    print("SWE_PREPARE stage=package_metadata", flush=True)
    if any(importlib.util.find_spec(name) is not None for name in ("torch", "ray", "tensordict")):
        raise RuntimeError("Use the isolated preparation venv without Torch/Ray/TensorDict")
    versions = {}
    for name in ("transformers", "datasets", "huggingface_hub", "tokenizers"):
        versions[name] = importlib.metadata.version(name)
    print("SWE_PREPARE stage=tokenizer", flush=True)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), local_files_only=True)
    tokens = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Inspect the repository and fix the issue."}],
        tokenize=True, add_generation_prompt=True, enable_thinking=True, preserve_thinking=True,
    )
    if not len(tokens):
        raise ValueError("Tokenizer produced no prompt tokens")
    print("SWE_PREPARE stage=dataset", flush=True)
    if not args.data_dir.exists():
        prepare(SimpleNamespace(data_dir=args.data_dir, revision=REVISION))
    manifest = check_data(args.data_dir)
    report = dict(python=sys.executable, versions=versions, dataset=manifest,
                  tokenizer=str(args.model_path), tokenizer_prompt_tokens=len(tokens),
                  runtime_imports_checked=False,
                  evaluation_performed=False, apptainer_agent_backend_ready=False)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    print("SWE_VISTA_DATA_AND_PYTHON_READY", args.report)


if __name__ == "__main__":
    main()
