"""Compare DiscoveryWorld runs on common graded tasks after checking protocol identity."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_scienceworld_pair import aggregate, load, protocol_check
from envs.discoveryworld_protocol import SCENARIOS


def audit(contextgraph, foldagent):
    cm, cr = load(contextgraph, "contextgraph")
    fm, fr = load(foldagent, "foldagent")
    if any(m.get("source", {}).get("benchmark") != "discoveryworld" for m in cm + fm):
        raise ValueError("Expected DiscoveryWorld manifests")
    protocol = protocol_check(cm, fm)
    common = sorted(t for t in cr.keys() & fr.keys() if cr[t]["status"] == fr[t]["status"] == "graded")
    def paired(ids):
        return {"count": len(ids), "contextgraph": aggregate([cr[t] for t in ids]),
                "foldagent": aggregate([fr[t] for t in ids]),
                "score_delta_contextgraph_minus_foldagent": sum(cr[t]["score"] - fr[t]["score"] for t in ids) / len(ids) if ids else None}
    return {"protocol": protocol, "common": paired(common), "common_task_ids": common,
            "all_completed": {"contextgraph": aggregate(list(cr.values())), "foldagent": aggregate(list(fr.values()))},
            "by_scenario": {name: paired([t for t in common if t.split('_')[1] == str(i)]) for i, name in enumerate(SCENARIOS)},
            "knowledge_score": None, "limits": "Text-only completion/procedural scores; knowledge evaluation not implemented. Unequal completed subsets are not a fair score comparison."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contextgraph", type=Path, required=True)
    parser.add_argument("--foldagent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.contextgraph, args.foldagent)
    with args.output.open("x", encoding="utf8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps({"protocol_matched": report["protocol"]["matched"], "common": report["common"]}, indent=2))
    sys.exit(0 if report["protocol"]["matched"] else 2)
