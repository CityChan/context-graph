"""Attach training document IDs by exact question; never alter prompts or validation data.

Labels JSON format: {"exact training question": ["docid", ...], ...}.
The caller must use training annotations from the same retriever corpus.
"""
import argparse
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.evidence_credit import gold_docids


def attach(rows, labels):
    result = []
    for row in rows:
        row = copy.deepcopy(row)
        extra = row["extra_info"]
        question = extra["query"]
        if question not in labels:
            raise ValueError(f"Missing training evidence labels for question: {question[:100]}")
        extra["graph_rpo_gold_docids"] = sorted(gold_docids({"graph_rpo_gold_docids": labels[question]}))
        result.append(row)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    import pandas as pd
    if args.check:
        rows = pd.read_parquet(args.check).to_dict("records")
        if not rows:
            raise ValueError("Empty training dataset")
        for row in rows:
            gold_docids(row["extra_info"])
        print(json.dumps({"training_rows": len(rows), "evidence_labels": "validated"}))
        return
    if not all((args.source, args.labels, args.output)):
        parser.error("Specify --check, or --source --labels --output")
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = attach(pd.read_parquet(args.source).to_dict("records"), json.loads(args.labels.read_text(encoding="utf8")))
    if not rows:
        raise ValueError("Empty training dataset")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(args.output, index=False)
    print(json.dumps({"training_rows": len(rows), "output": str(args.output)}))


if __name__ == "__main__":
    main()
