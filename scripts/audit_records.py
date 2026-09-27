"""Read persisted JSON/JSONL audit records with stable input ordering."""

import json
from pathlib import Path
from typing import Any, Iterable


def input_files(inputs: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for path in inputs:
        if path.is_dir():
            files.extend(sorted(path.rglob("*.jsonl")))
            files.extend(sorted(path.rglob("*.json")))
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(path)
    return list(dict.fromkeys(files))


def load_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(payload, dict):
        payload = payload.get("results") if "results" in payload else [payload]
    if not isinstance(payload, list):
        raise ValueError(f"{path} does not contain a JSON result list or JSONL records")
    return [record for record in payload if isinstance(record, dict)]
