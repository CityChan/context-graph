"""Check saved evidence-credit events; fail if the new backend was never exercised."""
import argparse
import json
import math
from pathlib import Path


def audit(paths):
    events, seen = [], set()
    def visit(value):
        if isinstance(value, dict):
            trace = value.get("graph_trace")
            if isinstance(trace, dict):
                identity = (str(value.get("gen_uid", "")), trace.get("final_hash"))
                if identity not in seen:
                    seen.add(identity)
                    events.extend(e for e in trace.get("events", []) if e.get("graph_rpo_credit_backend") == "evidence")
            for key, child in value.items():
                if key != "graph_trace":
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    for path in paths:
        for line in path.read_text(encoding="utf8").splitlines():
            if line.strip():
                visit(json.loads(line))
    for event in events:
        if not math.isfinite(event["graph_rpo_delta"]) or not isinstance(event.get("assistant_turn_index"), int):
            raise ValueError("Invalid evidence decision credit/span")
        if event.get("graph_rpo_outcome_gated") is not False:
            raise ValueError("Evidence credit unexpectedly outcome-gated")
        raw = (len(event['evidence_gained']) - len(event['evidence_lost'])) / event['evidence_gold_count']
        raw -= event['evidence_duplicate_penalty'] * event['duplicate_without_new_documents']
        expected = max(-event['graph_rpo_delta_max'], min(event['graph_rpo_delta_max'], raw / event['graph_rpo_delta_scale']))
        if not math.isclose(expected, event['graph_rpo_delta'], abs_tol=1e-8):
            raise ValueError("Evidence credit does not match recorded components")
    return {"decisions": len(events), "nonzero": sum(abs(e['graph_rpo_delta']) > 1e-12 for e in events),
            "branches": sum(e['op'] == 'branch' for e in events),
            "gained": sum(len(e['evidence_gained']) for e in events),
            "lost": sum(len(e['evidence_lost']) for e in events),
            "duplicate_without_new_documents": sum(e['duplicate_without_new_documents'] for e in events)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--require-nonzero", action="store_true")
    args = parser.parse_args()
    report = audit(args.paths)
    print(json.dumps(report))
    if not report['decisions'] or (args.require_nonzero and not report['nonzero']):
        raise SystemExit("Evidence credit smoke failed: no decisions/nonzero credit")


if __name__ == "__main__":
    main()
