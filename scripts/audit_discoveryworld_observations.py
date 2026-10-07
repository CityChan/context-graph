"""Offline, snapshot-local losslessness and size audit; never calls a model or simulator."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from envs.discoveryworld_observation import decode, dumps, encode


def audit(root, tokenizer=None):
    records = []
    for path in sorted(root.rglob("tools.jsonl")):
        record = {"file": str(path.relative_to(root)), "snapshots": 0,
                  "full_chars": 0, "compact_chars": 0}
        if tokenizer is not None:
            record.update(full_tokens=0, compact_tokens=0)
        with path.open(encoding="utf8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("event") not in ("reset", "step"):
                    continue
                ui = row.get("observation")
                if not isinstance(ui, dict) or "nearbyObjects" not in ui:
                    continue
                packed = encode(ui, "compact_v1")
                if decode(json.loads(dumps(packed, "compact_v1"))) != ui:
                    raise ValueError(f"Observation round-trip mismatch: {path}")
                # Includes the per-snapshot legend and dictionary, and action result.
                full, compact = ui, packed
                if row["event"] == "step":
                    full = {"action_result": row["result"], "ui": full}
                    compact = {"action_result": row["result"], "ui": compact}
                full, compact = dumps(full), dumps(compact, "compact_v1")
                record["snapshots"] += 1
                record["full_chars"] += len(full)
                record["compact_chars"] += len(compact)
                if tokenizer is not None:
                    record["full_tokens"] += len(tokenizer.encode(full, add_special_tokens=False))
                    record["compact_tokens"] += len(tokenizer.encode(compact, add_special_tokens=False))
        if record["snapshots"]:
            records.append(record)
    if not records:
        raise ValueError("No DiscoveryWorld UI observations found")
    totals = {key: sum(row[key] for row in records) for key in records[0] if key != "file"}
    unit = "tokens" if tokenizer is not None else "chars"
    totals["reduction_fraction"] = 1 - totals["compact_" + unit] / totals["full_" + unit]
    return {"round_trip_passed": True, "profile": "compact_v1",
            "tokenizer": getattr(tokenizer, "name_or_path", None), "totals": totals,
            "scope": "Serialized public observations only; not full prompts or model performance",
            "records": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--tokenizer", help="Local tokenizer snapshot; omit for character counts only")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    report = audit(args.root, tokenizer)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        with args.output.open("x", encoding="utf8") as stream:
            stream.write(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
