"""Convert existing BC-P train/test files to private-label GRAM RL smoke data."""
import json
from pathlib import Path

from agents.gram_memory import entity_key
from scripts.prepare_gram_data import sha256, write_json


def prepare_bcp(root, train_source, test_source, train_count=4, val_count=2, seed=42):
    import pandas as pd
    from scripts.eval_bcp_qwen38 import select_indices
    root = Path(root)
    sources = [Path(train_source), Path(test_source)]
    if sources[0].resolve() == sources[1].resolve():
        raise ValueError("BC-P train and test source must differ")
    frames = [pd.read_parquet(p).to_dict("records") for p in sources]
    question_sets = []
    for rows in frames:
        questions = []
        for row in rows:
            extra = row["extra_info"]
            if any(not isinstance(extra.get(k), str) or not extra[k].strip() for k in ("query", "answer")):
                raise ValueError("Missing BC-P question/reference")
            questions.append(entity_key(extra["query"]))
        if len(set(questions)) != len(questions):
            raise ValueError("Duplicate BC-P questions within a source")
        question_sets.append(set(questions))
    if question_sets[0] & question_sets[1]:
        raise ValueError("BC-P train/test question overlap")
    for source, rows, role, count in zip(sources, frames, ("train", "validation"), (train_count, val_count)):
        if count < 1 or len(rows) < count:
            raise ValueError("Insufficient BC-P rows for smoke")
        output = root / role
        if output.exists():
            raise ValueError("Use a fresh smoke data directory")
        indices = select_indices(len(rows), count, seed)
        prepared = []
        for i in indices:
            extra = rows[i]["extra_info"]
            task = {"task_id": f"bcp-{role}-{i}", "question": extra["query"], "documents": []}
            prepared.append({"data_source": f"gram/bcp/{role}", "ability": "GRAM-BCP",
                             "agent_name": "gram_agent", "prompt": [{"role": "user", "content": task["question"]}],
                             "extra_info": {"gram_task_json": json.dumps(task)},
                             "reward_model": {"style": "rule", "ground_truth": json.dumps([extra["answer"]])}})
        output.mkdir(parents=True)
        path = output / "data.parquet"
        pd.DataFrame(prepared).to_parquet(path, index=False)
        write_json(output / "manifest.json", {"benchmark": "bcp", "split": role,
                   "source_role": "bc_train" if role == "train" else "bc_test",
                   "source": str(source.resolve()), "source_sha256": sha256(source),
                   "indices": indices, "seed": seed, "count": count,
                   "files": {"data.parquet": sha256(path)}, "full_source_overlap_checked": True})
    from scripts.train_gram import check_data
    check_data(root / "train/data.parquet", root / "validation/data.parquet")
