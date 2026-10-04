"""Read an already prepared HF corpus without opening DatasetBuilder locks.

Only reading/concatenating Arrow datasets is allowed here: transformations that
write intermediate cache files would defeat use on a full or read-only volume.
"""
import json
from pathlib import Path


def load_cached_corpus(dataset_name, *, cache_root=None, snapshot=None):
    from datasets import Dataset, concatenate_datasets, config
    from datasets.naming import camelcase_to_snakecase, filenames_for_dataset_split

    namespace, name = dataset_name.split("/", 1)
    name = camelcase_to_snakecase(name)
    if snapshot is None:
        root = Path(cache_root or config.HF_DATASETS_CACHE)
        # Match the existing offline loader's latest-snapshot selection, scoped
        # to this corpus's default configuration. Never scan other datasets.
        base = root / f"{namespace}___{name}" / "default"
        candidates = [p.parent for p in base.glob("*/*/dataset_info.json")
                      if not p.parent.name.endswith(".incomplete")]
        if not candidates:
            raise FileNotFoundError(f"No prepared default corpus cache under {base}")
        snapshot = max(candidates, key=lambda p: p.stat().st_mtime)
    snapshot = Path(snapshot).resolve()
    info = json.loads((snapshot / "dataset_info.json").read_text(encoding="utf-8"))
    if info.get("dataset_name") != name or info.get("config_name") != "default":
        raise ValueError(f"Corpus cache identity mismatch for {dataset_name}: {snapshot}")
    train = info.get("splits", {}).get("train", {})
    expected_rows = train.get("num_examples")
    if not isinstance(expected_rows, int) or expected_rows <= 0:
        raise ValueError(f"Missing/nonpositive train row count in {snapshot}")
    lengths = train.get("shard_lengths")
    if lengths is not None and (not lengths or any(type(n) is not int or n < 0 for n in lengths)
                                or sum(lengths) != expected_rows):
        raise ValueError(f"Invalid train shard lengths in {snapshot}")
    paths = filenames_for_dataset_split(str(snapshot), name, "train", "arrow", lengths)
    shards = []
    for index, path in enumerate(paths):
        shard = Dataset.from_file(path)
        if lengths and len(shard) != lengths[index]:
            raise ValueError(f"Corpus shard row count mismatch: {path}")
        missing = {"docid", "url", "text"} - set(shard.column_names)
        if missing:
            raise ValueError(f"Corpus shard missing columns {sorted(missing)}: {path}")
        shards.append(shard)
    corpus = shards[0] if len(shards) == 1 else concatenate_datasets(shards)
    if len(corpus) != expected_rows:
        raise ValueError(f"Corpus train row count mismatch: {len(corpus)} != {expected_rows}")
    print("SEARCH_CORPUS_CACHE_READ_ONLY " + json.dumps({
        "dataset": dataset_name, "snapshot": str(snapshot), "rows": len(corpus),
        "shards": paths,
    }), flush=True)
    return corpus
