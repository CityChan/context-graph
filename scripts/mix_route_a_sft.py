"""Mix Route A with separately identified assisted explicit + thinking data.

Legacy message exports are re-tokenized, not claimed to be native-token
rollouts. Only explicitly marked assistant targets are supervised.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.exact_context import checksum, plain_messages, tokenizer_signature, validate_agent_context


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plain(value):
    """Normalize Arrow nested arrays without importing the training stack."""
    import numpy as np
    if isinstance(value, np.ndarray):
        return [plain(v) for v in value.tolist()]
    if isinstance(value, list):
        return [plain(v) for v in value]
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    return value


def legacy_segments(row, tokenizer, signature, max_length):
    """Re-render each marked answer with its saved prefix; never guess labels."""
    messages = row["messages"]
    if not messages or any(type(m.get("train")) is not bool for m in messages):
        raise ValueError("Legacy messages require explicit boolean train flags on every message")
    if any(m["train"] and m.get("role") != "assistant" for m in messages):
        raise ValueError("Only assistant messages may carry loss")
    results = []
    for i, message in enumerate(messages):
        if not message["train"]:
            continue
        prefix = plain_messages(messages[:i])
        target = plain_messages([message])[0]
        kwargs = dict(tools=row.get("tools") or [], tokenize=True, enable_thinking=True)
        prompt = list(tokenizer.apply_chat_template(prefix, add_generation_prompt=True, **kwargs))
        ids = list(tokenizer.apply_chat_template(prefix + [target], add_generation_prompt=False, **kwargs))
        if not prompt or ids[:len(prompt)] != prompt or len(ids) <= len(prompt):
            raise ValueError("Legacy target template is not prefix-compatible")
        # Templates may silently strip reasoning; reject rather than lose it.
        text = target.get("content")
        if not isinstance(text, str):
            raise ValueError("Legacy import supports text messages only")
        rendered = tokenizer.apply_chat_template(prefix + [target], tools=kwargs["tools"],
                                                tokenize=False, enable_thinking=True,
                                                add_generation_prompt=False)
        reasoning = target.get("reasoning_content")
        if text not in rendered or (reasoning and reasoning not in rendered):
            raise ValueError("Legacy template dropped answer or thinking content")
        context = {"version": 1, "input_ids": ids,
                   "loss_mask": [0] * len(prompt) + [1] * (len(ids) - len(prompt)),
                   "spans": [{"start": len(prompt), "end": len(ids),
                              "prefix_sha256": checksum(prompt),
                              "output_sha256": checksum(ids[len(prompt):]), "request_index": i}]}
        context["sha256"] = checksum(context)
        result = {"messages": [dict(m, train=False) for m in prefix] + [dict(target, train=True)],
                  "tools": kwargs["tools"], "agent_context": context, "tokenizer_sha256": signature}
        validate_agent_context(result, max_length, signature)
        results.append(result)
    if not results:
        raise ValueError("Legacy row has no supervised assistant answers")
    return results


def load_assisted(manifest_path, tokenizer, max_length):
    import pandas as pd
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    signature = tokenizer_signature(tokenizer)
    required = {"profile": "assisted_explicit_thinking_v1", "enable_thinking": True,
                "graph_action_protocol": "explicit_tools", "task_identity": "question_hash",
                "tokenizer_sha256": signature}
    if any(manifest.get(k) != v for k, v in required.items()):
        raise ValueError("Assisted source manifest must identify protocol, thinking, question hashes and tokenizer")
    if manifest.get("format") not in {"messages_train", "agent_context"}:
        raise ValueError("Assisted format must be messages_train or agent_context")
    buckets = {"train": [], "validation": []}
    sources = []
    for split in buckets:
        for entry in manifest["files"].get(split, []):
            source = (path.parent / entry["path"]).resolve()
            if sha_file(source) != entry["sha256"]:
                raise ValueError(f"Assisted source hash mismatch: {source}")
            sources.append({"split": split, "path": str(source), "sha256": entry["sha256"]})
            for index, raw in enumerate(pd.read_parquet(source).to_dict("records")):
                row = plain(raw)
                task = row.get("question_hash") or row.get("task_id")
                if not isinstance(task, str) or len(task) != 64 or any(c not in "0123456789abcdef" for c in task):
                    raise ValueError("Assisted rows need the same canonical question hash as Route A")
                if row.get("enable_thinking") is not True:
                    raise ValueError("Assisted row is not marked thinking-enabled")
                if row.get("split") not in (None, split):
                    raise ValueError("Assisted row split disagrees with source file split")
                if manifest["format"] == "agent_context":
                    validate_agent_context(row, max_length, signature)
                    parts = [copy.deepcopy(row)]
                else:
                    parts = legacy_segments(row, tokenizer, signature, max_length)
                for part in parts:
                    part.update(task_id=task, split=split, sft_profile="route_a_mixed_v1",
                                enable_thinking=True, tools=row.get("tools") or [],
                                source_family="assisted_explicit_thinking",
                                context_origin=("legacy_messages_retokenized" if manifest["format"] == "messages_train"
                                                else "source_agent_context"),
                                graph_action_protocol="explicit_tools",
                                source_row=index, source_file_sha256=entry["sha256"],
                                source_manifest_sha256=sha_file(path),
                                # Exact masks already specify targets; do not re-render them in the dataset.
                                train_last_response_only=False)
                    buckets[split].append(part)
    if not buckets["train"]:
        raise ValueError("No assisted thinking training rows")
    return buckets, sources


def mix(args, tokenizer):
    import pandas as pd
    signature = tokenizer_signature(tokenizer)
    root, output = Path(args.route_a), Path(args.output)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("profile") != "route_a_v1" or manifest.get("tokenizer_sha256") != signature:
        raise ValueError("Route A profile/tokenizer mismatch")
    if not math.isfinite(args.assisted_ratio) or args.assisted_ratio <= 0:
        raise ValueError("assisted_ratio must be positive and finite")
    buckets = {"train": [], "validation": []}
    sources = []
    for split in buckets:
        source = root / f"{split}.parquet"
        expected = manifest["files_sha256"].get(source.name)
        if not expected or sha_file(source) != expected:
            raise ValueError(f"Missing or modified Route A {split} export")
        sources.append({"split": split, "path": str(source.resolve()), "sha256": expected})
        for raw in pd.read_parquet(source).to_dict("records"):
            row = plain(raw)
            validate_agent_context(row, args.max_length, signature)
            if row["split"] != split or row["enable_thinking"] is not True:
                raise ValueError("Route A split/thinking mismatch")
            row.update(source_family="route_a", context_origin="native_token_ids",
                       sft_profile="route_a_mixed_v1", train_last_response_only=False)
            buckets[split].append(row)
    assisted, assisted_sources = load_assisted(args.assisted_manifest, tokenizer, args.max_length)
    # Check all supplied rows BEFORE downsampling so the ratio cannot hide leakage.
    train_tasks = {r["task_id"] for r in buckets["train"] + assisted["train"]}
    val_tasks = {r["task_id"] for r in buckets["validation"] + assisted["validation"]}
    if train_tasks & val_tasks:
        raise ValueError("Task overlap across train/validation; reconcile source splits before mixing")
    def target_keys(rows):
        return {(s["prefix_sha256"], s["output_sha256"])
                for r in rows for s in r["agent_context"]["spans"]}
    if target_keys(buckets["train"] + assisted["train"]) & target_keys(buckets["validation"] + assisted["validation"]):
        raise ValueError("Supervised response overlap across train/validation, even with different task IDs")
    seen = set()
    duplicates = Counter()
    def unique(rows):
        accepted = []
        for row in rows:
            context = row["agent_context"]
            spans = context["spans"]
            keys = [(s["prefix_sha256"], s["output_sha256"]) for s in spans]
            keep = [i for i, key in enumerate(keys) if key not in seen]
            if len(set(keys)) != len(keys):
                raise ValueError("Duplicate supervised response within a segment")
            duplicates[row["source_family"]] += len(keys) - len(keep)
            if not keep:
                continue
            row = copy.deepcopy(row)
            context = row["agent_context"]
            context["spans"] = [spans[i] for i in keep]
            context["loss_mask"] = [0] * len(context["input_ids"])
            for span in context["spans"]:
                context["loss_mask"][span["start"]:span["end"]] = [1] * (span["end"] - span["start"])
            target_index = 0
            for message in row["messages"]:
                if message.get("train") is True:
                    message["train"] = target_index in keep
                    target_index += 1
            context["sha256"] = checksum({k: v for k, v in context.items() if k != "sha256"})
            validate_agent_context(row, args.max_length, signature)
            seen.update(keys)
            accepted.append(row)
        return accepted
    for split in buckets:
        buckets[split] = unique(buckets[split])
        assisted[split] = unique(assisted[split])
    limit = math.floor(args.assisted_ratio * len(buckets["train"]))
    selected = sorted(assisted["train"], key=lambda r: checksum(
        [args.seed, r["task_id"], r["agent_context"]["sha256"]]))[:limit]
    if not selected:
        raise ValueError("No distinct assisted rows selected; adjust ratio or inspect duplicates")
    buckets["train"] += selected
    buckets["validation"] += assisted["validation"]
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use an empty mixture output directory")
    output.mkdir(parents=True, exist_ok=True)
    hashes, stats = {}, {}
    for split, rows in buckets.items():
        rows.sort(key=lambda r: checksum([args.seed, r["task_id"], r["agent_context"]["sha256"]]))
        target = output / f"{split}.parquet"
        pd.DataFrame(rows).to_parquet(target, index=False)
        hashes[target.name] = sha_file(target)
        stats[split] = {family: {"rows": sum(r["source_family"] == family for r in rows),
                                "loss_tokens": sum(sum(r["agent_context"]["loss_mask"]) for r in rows
                                                   if r["source_family"] == family)}
                        for family in ("route_a", "assisted_explicit_thinking")}
    report = {"profile": "route_a_mixed_v1", "settings": vars(args),
              "tokenizer_sha256": signature, "sources": sources + assisted_sources,
              "assisted_manifest_sha256": sha_file(args.assisted_manifest),
              "counts": stats, "duplicate_targets_removed": dict(duplicates),
              "assisted_train_available": len(assisted["train"]),
              "files_sha256": hashes, "training_ready": True,
              "scope": "Mixed offline SFT; source protocols preserved; legacy text is re-tokenized, not native rollout evidence"}
    (output / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-a")
    parser.add_argument("--assisted-manifest", required=True)
    parser.add_argument("--output")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--register-assisted", action="store_true",
                        help="Register existing curated assisted explicit thinking files; does not curate or relabel data")
    parser.add_argument("--assisted-train", nargs="+")
    parser.add_argument("--assisted-validation", nargs="*", default=[])
    parser.add_argument("--assisted-format", choices=["messages_train", "agent_context"], default="messages_train")
    parser.add_argument("--assisted-ratio", type=float, default=1.0,
                        help="Maximum assisted rows per Route A training row; no repetition (default 1)")
    parser.add_argument("--max-length", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    if args.register_assisted:
        if not args.assisted_train or args.route_a or args.output:
            parser.error("Registration requires --assisted-train, without --route-a or --output")
        manifest = {"profile": "assisted_explicit_thinking_v1", "enable_thinking": True,
                    "graph_action_protocol": "explicit_tools", "task_identity": "question_hash",
                    "tokenizer_sha256": tokenizer_signature(tokenizer),
                    "source_tokenizer": str(Path(args.tokenizer).resolve()), "format": args.assisted_format,
                    "provenance": "operator-designated existing curated assisted explicit thinking export",
                    "files": {split: [{"path": str(Path(p).resolve()), "sha256": sha_file(p)} for p in paths]
                              for split, paths in (("train", args.assisted_train),
                                                   ("validation", args.assisted_validation))}}
        with Path(args.assisted_manifest).open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(manifest, indent=2))
        print("Registered source file identities; mixing will validate rows, masks and split separation.")
    else:
        if not args.route_a or not args.output:
            parser.error("Mixing requires --route-a and --output")
        print(json.dumps(mix(args, tokenizer), indent=2))


if __name__ == "__main__":
    main()
