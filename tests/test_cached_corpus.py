"""Prepared corpus reads must survive a full cache volume, without data changes."""
import ast
import errno
import json
import os
from pathlib import Path

import pyarrow as pa
import pytest
from datasets.packaged_modules.cache.cache import Cache
from filelock import FileLock

from envs.cached_corpus import load_cached_corpus


CORPUS = "Tevatron/browsecomp-plus-corpus"
NAME = "browsecomp-plus-corpus"


def prepared_cache(root, *, sharded=True, revision="revision-a"):
    snapshot = root / "Tevatron___browsecomp-plus-corpus/default/0.0.0" / revision
    snapshot.mkdir(parents=True)
    rows = [
        {"docid": "0042", "url": "https://example.org/a", "text": "原始 text\n" + "word " * 15010,
         "title": "Do not prepend this title"},
        {"docid": "9", "url": "https://example.org/b", "text": "Second document", "title": "Title"},
    ]
    chunks = [[row] for row in rows] if sharded else [rows]
    for index, chunk in enumerate(chunks):
        suffix = f"-{index:05d}-of-{len(chunks):05d}" if sharded else ""
        with (snapshot / f"{NAME}-train{suffix}.arrow").open("wb") as handle:
            table = pa.Table.from_pylist(chunk)
            with pa.ipc.new_stream(handle, table.schema) as writer:
                writer.write_table(table)
    info = {"dataset_name": NAME, "config_name": "default", "version": "0.0.0",
            "splits": {"train": {"name": "train", "num_examples": len(rows),
                                  "num_bytes": 0, "shard_lengths": [1, 1] if sharded else None}}}
    (snapshot / "dataset_info.json").write_text(json.dumps(info), encoding="utf-8")
    return snapshot, rows


def corpus_mapping(loader):
    """Run the server's actual text/index conversion without loading GPU models."""
    source = Path(__file__).resolve().parents[1] / "envs/search_server.py"
    nodes = [n for n in ast.parse(source.read_text(encoding="utf-8")).body
             if isinstance(n, ast.FunctionDef) and n.name in {"keep_first_n_words", "load_corpus"}]
    import re
    namespace = dict(os=os, re=re, LOCAL_CORPUS_PARQUET=None, CORPUS_DATASET=CORPUS,
                     load_dataset=lambda *a, **kw: loader(),
                     load_cached_corpus=lambda *a, **kw: loader())
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    return namespace["load_corpus"]()


@pytest.mark.parametrize("sharded", [False, True])
def test_full_disk_loader_matches_normal_builder_without_writes(tmp_path, monkeypatch, sharded):
    snapshot, rows = prepared_cache(tmp_path, sharded=sharded)
    builder = Cache(cache_dir=str(tmp_path), repo_id=CORPUS, dataset_name=NAME,
                    config_name="default", version="0.0.0", hash="revision-a")
    baseline = builder.as_dataset(split="train")
    assert list(baseline) == rows
    expected = corpus_mapping(lambda: baseline)
    before = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}

    def full_disk(*args, **kwargs):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(FileLock, "acquire", full_disk)
    with pytest.raises(OSError, match="No space left"):
        Cache(cache_dir=str(tmp_path), repo_id=CORPUS, dataset_name=NAME,
              config_name="default", version="0.0.0", hash="revision-a")
    monkeypatch.setenv("SEARCH_CORPUS_CACHE_READ_ONLY", "1")
    actual = load_cached_corpus(CORPUS, cache_root=tmp_path)
    assert list(actual) == rows
    assert corpus_mapping(lambda: actual) == expected
    assert {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before


def test_latest_snapshot_and_explicit_pin(tmp_path):
    first, _ = prepared_cache(tmp_path, revision="first")
    second, _ = prepared_cache(tmp_path, revision="second")
    os.utime(first, (100, 100))
    os.utime(second, (200, 200))
    dataset = load_cached_corpus(CORPUS, cache_root=tmp_path)
    assert all(str(second) in entry["filename"] for entry in dataset.cache_files)
    pinned = load_cached_corpus(CORPUS, snapshot=first)
    assert all(str(first) in entry["filename"] for entry in pinned.cache_files)


@pytest.mark.parametrize("failure", ["missing_shard", "wrong_rows", "wrong_shard_rows", "wrong_identity", "missing_columns"])
def test_incomplete_or_wrong_cache_fails_instead_of_partial_corpus(tmp_path, failure):
    snapshot, _ = prepared_cache(tmp_path)
    info_path = snapshot / "dataset_info.json"
    info = json.loads(info_path.read_text())
    if failure == "missing_shard":
        (snapshot / f"{NAME}-train-00001-of-00002.arrow").unlink()
    elif failure == "wrong_rows":
        info["splits"]["train"]["num_examples"] = 3
    elif failure == "wrong_shard_rows":
        info["splits"]["train"]["shard_lengths"] = [0, 2]
    elif failure == "wrong_identity":
        info["dataset_name"] = "other-corpus"
    else:
        table = pa.Table.from_pylist([{"docid": "0042"}])
        with (snapshot / f"{NAME}-train-00000-of-00002.arrow").open("wb") as handle:
            with pa.ipc.new_stream(handle, table.schema) as writer:
                writer.write_table(table)
    info_path.write_text(json.dumps(info), encoding="utf-8")
    with pytest.raises((ValueError, FileNotFoundError)):
        load_cached_corpus(CORPUS, cache_root=tmp_path)


def test_missing_cache_does_not_download(tmp_path):
    with pytest.raises(FileNotFoundError, match="No prepared"):
        load_cached_corpus(CORPUS, cache_root=tmp_path)
