"""Detect the observed token-zero generation failure without changing outputs."""
import json
from pathlib import Path


MIN_DEGENERATE_RUN = 20


def has_degenerate_run(output_ids):
    run = 0
    for token in output_ids:
        run = run + 1 if token == 0 else 0
        if run >= MIN_DEGENERATE_RUN:
            return True
    return False


def _counts(total, bad, *, unaudited=0):
    return {"model_requests": total, "degenerate_requests": bad,
            "degenerate_request_rate": bad / total if total else None,
            "generation_quality_passed": False if bad * 100 > total else
                                         None if not total or unaudited else True,
            "generation_unaudited_records": unaudited}


def degeneration_stats(paths):
    """Scan explicit request files once; never concatenate requests or input IDs.

    Callers select current attempts. Missing/malformed supplied files fail loudly;
    no supplied files means no audit evidence, not a clean generation result.
    """
    total = bad = 0
    for path in paths:
        path = Path(path)
        with path.open(encoding="utf-8") as handle:
            for lineno, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    ids = row["output_ids"]
                    if not isinstance(ids, list) or any(type(t) is not int or t < 0 for t in ids):
                        raise ValueError("output_ids must be nonnegative integers")
                except (ValueError, KeyError, TypeError) as exc:
                    raise ValueError(f"Invalid generation audit at {path}:{lineno}: {exc}") from exc
                total += 1
                bad += has_degenerate_run(ids)
    return _counts(total, bad)


def combine_degeneration_stats(records):
    """Aggregate saved current-attempt counts without rescanning growing logs."""
    total = bad = unaudited = 0
    for record in records:
        stats = record.get("generation_audit")
        if stats is None or not stats["model_requests"]:
            unaudited += 1
        else:
            total += stats["model_requests"]
            bad += stats["degenerate_requests"]
    return _counts(total, bad, unaudited=unaudited)


def require_generation_quality(stats):
    # Integer comparison keeps exactly 1% admissible, including 1/100 and 2/200.
    if stats["degenerate_requests"] * 100 > stats["model_requests"]:
        raise RuntimeError(
            "Generation degeneration exceeds 1% of model requests "
            f"({stats['degenerate_requests']}/{stats['model_requests']}); "
            "do not treat this evaluation as a clean score. Inspect token IDs and server execution mode.")
