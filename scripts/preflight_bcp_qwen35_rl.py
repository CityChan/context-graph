"""Offline dependency/architecture preflight, not a distributed training smoke."""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


def import_in_fresh_process(module: str, timeout=180):
    """Do not let a previous probe's partially imported modules mask a cycle."""
    result = subprocess.run(
        [sys.executable, "-c", f"import importlib; importlib.import_module({module!r})"],
        capture_output=True, text=True, timeout=timeout,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    if result.returncode:
        raise RuntimeError(f"Fresh import exited {result.returncode}\n{result.stdout}\n{result.stderr}")
    return {"fresh_process": True, "returncode": result.returncode}


def check_checkpoint(root: Path) -> dict:
    config = json.loads((root / "config.json").read_text(encoding="utf8"))
    if config.get("model_type") != "qwen3_5":
        raise ValueError(f"Expected Qwen3.5, got {config.get('model_type')}")
    index = root / "model.safetensors.index.json"
    if index.exists():
        names = set(json.loads(index.read_text(encoding="utf8"))["weight_map"].values())
    else:
        names = {"model.safetensors"}
    if not names:
        raise ValueError("Empty checkpoint weight index")
    for name in names:
        path = (root / name).resolve()
        # HF snapshot files may legitimately be symlinks into ../blobs.
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing or empty checkpoint shard: {name}")
    return {"model_type": config["model_type"], "architectures": config.get("architectures"),
            "weight_shards": len(names)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"model_path": str(args.model_path), "checks": {}, "errors": {}, "versions": {}, "tracebacks": {}}

    def check(label, fn):
        try:
            report["checks"][label] = fn() or "ok"
            print(f"PASS {label}", flush=True)
        except Exception as exc:
            report["errors"][label] = f"{type(exc).__name__}: {exc}"
            report["tracebacks"][label] = traceback.format_exc()
            print(f"FAIL {label}: {type(exc).__name__}: {exc}", flush=True)

    for package in ("torch", "transformers", "vllm", "ray", "flash-attn", "hydra-core", "tensordict"):
        try:
            report["versions"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report["versions"][package] = "not installed"
    check("checkpoint", lambda: check_checkpoint(args.model_path))

    def model_classes():
        from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer
        config = AutoConfig.from_pretrained(args.model_path, local_files_only=True)
        model_cls = AutoModelForImageTextToText._model_mapping[type(config)]
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
        ids = tokenizer.apply_chat_template([{"role": "user", "content": "Say OK."}],
            tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=True)
        assert isinstance(ids, list) and ids and all(isinstance(i, int) for i in ids), "Expected a nonempty list of token IDs"
        return {"model_class": model_cls.__name__, "tokenizer": type(tokenizer).__name__}

    check("HF model class and tokenizer", model_classes)
    for module in ("scripts.train_fold", "scripts.train_graph", "verl.workers.fsdp_workers",
                   "verl.workers.rollout.vllm_rollout.vllm_async_server"):
        check(module, lambda module=module: import_in_fresh_process(module))

    def structured_sampling():
        from vllm import SamplingParams, sampling_params
        from agents.structured_outputs import build_vllm_structured_sampling_kwargs
        schema = {"json": {"type": "object", "properties": {"action": {"const": "pass"}},
                           "required": ["action"], "additionalProperties": False}}
        SamplingParams(max_tokens=32, **build_vllm_structured_sampling_kwargs(schema, sampling_params))

    check("vLLM controller sampling API", structured_sampling)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf8")
    if report["errors"]:
        raise SystemExit("RL dependency preflight failed. See preflight.json; eval-server success does not validate the VERL training stack.")
    print("RL dependency preflight passed; distributed update/weight sync remain unverified.")


if __name__ == "__main__":
    main()
