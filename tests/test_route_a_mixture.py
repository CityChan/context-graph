import copy
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from agents.exact_context import checksum, segments_from_requests, tokenizer_signature
from scripts import mix_route_a_sft as mixer
from tests.test_route_a_sft import Tokenizer, record


def legacy_row(task="legacy train"):
    return {"task_id": checksum(task), "enable_thinking": True, "tools": [],
            "messages": [{"role": "user", "content": "rewritten evidence " + task, "train": False},
                         {"role": "assistant", "content": "historical answer", "train": False},
                         {"role": "user", "content": "maintenance checkpoint", "train": False},
                         {"role": "assistant", "content": "<think>keep source</think> merge", "train": True},
                         {"role": "user", "content": "continue", "train": False},
                         {"role": "assistant", "content": "<think>solve</think> final", "train": True}]}


def inputs(tmp_path):
    tok = Tokenizer()
    root = tmp_path / "route"
    root.mkdir()
    hashes = {}
    for split in ("train", "validation"):
        rows = []
        for i in range(2):
            req = record([{"role": "user", "content": f"route {split} {i}"}], "<think>act</think> search")
            row = segments_from_requests([req], 65536, tokenizer_signature(tok))[0]
            row.update(task_id=checksum(f"route {split} {i}"), split=split, tools=[],
                       enable_thinking=True, sft_profile="route_a_v1", graph_action_protocol="controller")
            rows.append(row)
        file = root / f"{split}.parquet"
        pd.DataFrame(rows).to_parquet(file, index=False)
        hashes[file.name] = mixer.sha_file(file)
    (root / "manifest.json").write_text(json.dumps(
        {"profile": "route_a_v1", "tokenizer_sha256": tokenizer_signature(tok), "files_sha256": hashes}))
    files = {}
    for split in ("train", "validation"):
        file = tmp_path / f"legacy-{split}.parquet"
        pd.DataFrame([legacy_row("legacy " + split)]).to_parquet(file, index=False)
        files[split] = [{"path": file.name, "sha256": mixer.sha_file(file)}]
    manifest = {"profile": "assisted_explicit_thinking_v1", "enable_thinking": True,
                "graph_action_protocol": "explicit_tools", "task_identity": "question_hash",
                "tokenizer_sha256": tokenizer_signature(tok), "format": "messages_train", "files": files}
    path = tmp_path / "assisted.json"
    path.write_text(json.dumps(manifest))
    return SimpleNamespace(route_a=str(root), assisted_manifest=str(path), output=str(tmp_path / "mixed"),
                           tokenizer="fixture", assisted_ratio=1.0, max_length=65536, seed=42), tok


def test_legacy_masks_history_preserves_thinking_and_splits_targets():
    tok = Tokenizer()
    rows = mixer.legacy_segments(legacy_row(), tok, tokenizer_signature(tok), 65536)
    assert len(rows) == 2
    for i, row in enumerate(rows):
        context = row["agent_context"]
        supervised = tok.decode([v for v, m in zip(context["input_ids"], context["loss_mask"]) if m])
        assert supervised == legacy_row()["messages"][3 + 2 * i]["content"] + "\0"
        assert "historical answer" not in supervised
        assert sum(m["train"] for m in row["messages"]) == 1
    # The earlier target becomes history for the second, never supervised twice.
    assert rows[1]["messages"][3]["train"] is False


def test_mixture_roundtrip_through_training_dataset(tmp_path):
    from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
    args, tok = inputs(tmp_path)
    report = mixer.mix(args, tok)
    assert report["counts"]["train"]["route_a"]["rows"] == 2
    assert report["counts"]["train"]["assisted_explicit_thinking"]["rows"] == 2
    file = str(tmp_path / "mixed" / "train.parquet")
    rows = mixer.plain(pd.read_parquet(file).to_dict("records"))
    assert {r["graph_action_protocol"] for r in rows} == {"controller", "explicit_tools"}
    assert {r["context_origin"] for r in rows} == {"native_token_ids", "legacy_messages_retokenized"}
    ds = MultiTurnSFTDataset(file, tok, config={"max_length": 65536, "truncation": "error",
                                               "pad_mode": "no_padding", "loss_mask_mode": "agent_context"})
    for i, row in enumerate(rows):
        item = ds[i]
        assert item["input_ids"].tolist() == row["agent_context"]["input_ids"]
        assert item["loss_mask"].tolist() == row["agent_context"]["loss_mask"]
    ds.loss_mask_mode = "per_message"
    with pytest.raises(ValueError, match="requires loss_mask_mode"):
        ds[0]


def rewrite_assisted(args, transform):
    path = mixer.Path(args.assisted_manifest)
    manifest = json.loads(path.read_text())
    file = path.parent / manifest["files"]["train"][0]["path"]
    rows = mixer.plain(pd.read_parquet(file).to_dict("records"))
    transform(rows)
    pd.DataFrame(rows).to_parquet(file, index=False)
    manifest["files"]["train"][0]["sha256"] = mixer.sha_file(file)
    path.write_text(json.dumps(manifest))


def test_reject_cross_source_task_leakage_even_when_downsampling(tmp_path):
    args, tok = inputs(tmp_path)
    args.assisted_ratio = 0.5
    rewrite_assisted(args, lambda rows: rows[0].update(task_id=checksum("route validation 0")))
    with pytest.raises(ValueError, match="overlap"):
        mixer.mix(args, tok)
    assert not mixer.Path(args.output).exists()


def test_source_hash_and_tokenizer_and_missing_flags_fail_closed(tmp_path):
    args, tok = inputs(tmp_path)
    tok.chat_template = "wrong template"
    with pytest.raises(ValueError, match="tokenizer"):
        mixer.mix(args, tok)
    tok = Tokenizer()
    (tmp_path / "legacy-train.parquet").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        mixer.mix(args, tok)
    row = legacy_row()
    del row["messages"][1]["train"]
    with pytest.raises(ValueError, match="train flags"):
        mixer.legacy_segments(row, tok, tokenizer_signature(tok), 65536)
    with pytest.raises(ValueError, match="truncation"):
        mixer.legacy_segments(legacy_row(), tok, tokenizer_signature(tok), 3)


def test_ratio_is_deterministic_and_does_not_repeat(tmp_path):
    args, tok = inputs(tmp_path)
    args.assisted_ratio = 0.5
    report = mixer.mix(args, tok)
    assert report["counts"]["train"]["assisted_explicit_thinking"]["rows"] == 1
    first = mixer.sha_file(mixer.Path(args.output) / "train.parquet")
    args.output = str(tmp_path / "second")
    mixer.mix(args, tok)
    assert first == mixer.sha_file(mixer.Path(args.output) / "train.parquet")
    args.output = str(tmp_path / "third")
    args.assisted_ratio = 100
    report = mixer.mix(args, tok)
    assert report["counts"]["train"]["assisted_explicit_thinking"]["rows"] == 2


def test_duplicate_legacy_targets_do_not_increase_weight(tmp_path):
    args, tok = inputs(tmp_path)
    rewrite_assisted(args, lambda rows: rows.append(copy.deepcopy(rows[0])))
    report = mixer.mix(args, tok)
    assert report["duplicate_targets_removed"]["assisted_explicit_thinking"] == 2
    assert report["counts"]["train"]["assisted_explicit_thinking"]["rows"] == 2


def test_native_assisted_import_preserves_original_masks(tmp_path):
    args, tok = inputs(tmp_path)
    path = mixer.Path(args.assisted_manifest)
    manifest = json.loads(path.read_text())
    manifest["format"] = "agent_context"
    for split in ("train", "validation"):
        file = tmp_path / f"legacy-{split}.parquet"
        row = mixer.legacy_segments(legacy_row(), tok, tokenizer_signature(tok), 65536)[0]
        row.update(task_id=checksum("native " + split), enable_thinking=True)
        # Give validation different evidence to avoid duplicate labels.
        if split == "validation":
            row = mixer.legacy_segments(legacy_row(), tok, tokenizer_signature(tok), 65536)[1]
            row.update(task_id=checksum("native validation"), enable_thinking=True)
        pd.DataFrame([row]).to_parquet(file, index=False)
        manifest["files"][split][0]["sha256"] = mixer.sha_file(file)
    path.write_text(json.dumps(manifest))
    report = mixer.mix(args, tok)
    assert report["counts"]["train"]["assisted_explicit_thinking"]["rows"] == 1


def test_same_supervised_response_under_different_task_ids_cannot_leak(tmp_path):
    args, tok = inputs(tmp_path)
    def change(rows):
        row = legacy_row("legacy validation")
        row["task_id"] = checksum("different task identifier")
        rows[0] = row
    rewrite_assisted(args, change)
    with pytest.raises(ValueError, match="Supervised response overlap"):
        mixer.mix(args, tok)


def test_template_that_strips_thinking_is_rejected():
    class StrippingTokenizer(Tokenizer):
        def apply_chat_template(self, messages, **kwargs):
            messages = copy.deepcopy(messages)
            for message in messages:
                message["content"] = message["content"].replace("<think>keep source</think>", "")
            return super().apply_chat_template(messages, **kwargs)
    tok = StrippingTokenizer()
    with pytest.raises(ValueError, match="dropped answer or thinking"):
        mixer.legacy_segments(legacy_row(), tok, tokenizer_signature(tok), 65536)


def test_partial_duplicate_native_segment_masks_only_repeated_target(tmp_path):
    args, tok = inputs(tmp_path)
    path = mixer.Path(args.assisted_manifest)
    manifest = json.loads(path.read_text())
    manifest["format"] = "agent_context"
    req1 = record([{"role": "user", "content": "first"}], "<think>one</think> a")
    req2 = record(req1["messages"] + [{"role": "assistant", "content": req1["response"]},
                                     {"role": "user", "content": "next"}], "<think>two</think> b")
    short = segments_from_requests([req1], 65536, tokenizer_signature(tok))[0]
    long = segments_from_requests([req1, req2], 65536, tokenizer_signature(tok))[0]
    rows = [short, long]
    for row in rows:
        row.update(task_id=checksum("same task"), enable_thinking=True)
    file = tmp_path / "legacy-train.parquet"
    pd.DataFrame(rows).to_parquet(file, index=False)
    manifest["files"]["train"][0]["sha256"] = mixer.sha_file(file)
    manifest["files"]["validation"] = []
    path.write_text(json.dumps(manifest))
    report = mixer.mix(args, tok)
    assert report["duplicate_targets_removed"]["assisted_explicit_thinking"] == 1
    mixed = mixer.plain(pd.read_parquet(mixer.Path(args.output) / "train.parquet").to_dict("records"))
    targets = [s for r in mixed if r["source_family"] == "assisted_explicit_thinking"
               for s in r["agent_context"]["spans"]]
    assert len(targets) == 2
    assert {s["output_sha256"] for s in targets} == {checksum(req1["output_ids"]), checksum(req2["output_ids"])}


def test_registration_and_mix_cli(tmp_path, monkeypatch, capsys):
    import sys
    import transformers
    args, tok = inputs(tmp_path)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **kw: tok)
    registered = tmp_path / "registered.json"
    monkeypatch.setattr(sys, "argv", ["mix", "--register-assisted", "--assisted-manifest", str(registered),
                                    "--tokenizer", "fixture", "--assisted-train", str(tmp_path / "legacy-train.parquet"),
                                    "--assisted-validation", str(tmp_path / "legacy-validation.parquet")])
    mixer.main()
    assert json.loads(registered.read_text())["files"]["train"][0]["sha256"] == mixer.sha_file(
        tmp_path / "legacy-train.parquet")
    capsys.readouterr()
    monkeypatch.setattr(sys, "argv", ["mix", "--assisted-manifest", str(registered),
                                    "--tokenizer", "fixture", "--route-a", args.route_a, "--output", args.output])
    mixer.main()
    assert json.loads(capsys.readouterr().out)["counts"]["train"]["assisted_explicit_thinking"]["rows"] == 2
