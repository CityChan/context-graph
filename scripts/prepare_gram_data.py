"""Convert local document-QA releases without oracle filtering or reordering evidence."""
import argparse
import hashlib
import json
from pathlib import Path

from agents.gram_memory import DocumentStream


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def convert(row, benchmark, evidence_root=None):
    identity = row.get("_id", row.get("id", row.get("QuestionId", row.get("task_id"))))
    if identity is None:
        raise ValueError("Missing task ID")
    question = row.get("question", row.get("Question"))
    if benchmark in {"hotpotqa", "2wiki"}:
        documents = [{"id": str(i), "title": title, "text": " ".join(sentences)}
                     for i, (title, sentences) in enumerate(row["context"])]
        answers = [row["answer"]]
    elif benchmark == "musique":
        if row.get("answerable", True) is not True:
            raise ValueError("Use the answerable MuSiQue split; unanswerable examples are not silently dropped")
        documents = [{"id": str(i), "title": p["title"], "text": p["paragraph_text"]}
                     for i, p in enumerate(row["paragraphs"])]
        answers = [row["answer"], *row.get("answer_aliases", [])]
    elif benchmark == "triviaqa":
        if evidence_root is None:
            raise ValueError("Native TriviaQA requires --evidence-root (containing wikipedia/ and web/)")
        root = Path(evidence_root).resolve()
        documents = []
        for group, folder in (("EntityPages", "wikipedia"), ("SearchResults", "web")):
            for page in row.get(group, []):
                path = (root / folder / page["Filename"]).resolve()
                if not path.is_relative_to(root / folder):
                    raise ValueError("Evidence path escapes its dataset directory")
                documents.append({"id": str(len(documents)), "title": page.get("Title", page["Filename"]),
                                  "text": path.read_text(encoding="utf-8")})
        answers = [row["Answer"]["Value"], *row["Answer"].get("Aliases", [])]
    else:
        documents, answers = row["documents"], row["answers"]
    DocumentStream(question, documents)
    if not isinstance(answers, list) or not answers or any(not isinstance(a, str) or not a.strip() for a in answers):
        raise ValueError("Cannot evaluate unlabeled data or empty references")
    public = {"task_id": f"{benchmark}:{identity}", "question": question, "documents": documents}
    return public, list(dict.fromkeys(answers))


def prepare(source, output, benchmark, split, evidence_root=None, parquet=False):
    source, output = Path(source), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be new or empty; prepared datasets are immutable")
    text = source.read_text(encoding="utf-8")
    rows = [json.loads(line) for line in text.splitlines() if line.strip()] if source.suffix == ".jsonl" else json.loads(text)
    if isinstance(rows, dict):
        rows = rows["Data"]
    tasks, references = [], {}
    for row in rows:
        task, answers = convert(row, benchmark, evidence_root)
        if task["task_id"] in references:
            raise ValueError("Duplicate task ID")
        tasks.append(task)
        references[task["task_id"]] = answers
    if not tasks:
        raise ValueError("No tasks")
    write_json(output / "tasks.json", tasks)
    write_json(output / "references.json", references)
    files = {name: sha256(output / name) for name in ("tasks.json", "references.json")}
    if parquet:
        import pandas as pd
        # JSON strings avoid Arrow turning nested dictionaries into arrays/unions.
        frame = pd.DataFrame([{
            "data_source": f"gram/{benchmark}/{split}", "ability": "GRAM",
            "agent_name": "gram_agent", "prompt": [{"role": "user", "content": t["question"]}],
            "extra_info": {"gram_task_json": json.dumps(t, ensure_ascii=False)},
            "reward_model": {"style": "rule", "ground_truth": json.dumps(references[t["task_id"]])},
        } for t in tasks])
        frame.to_parquet(output / "data.parquet", index=False)
        files["data.parquet"] = sha256(output / "data.parquet")
    manifest = {"benchmark": benchmark, "split": split, "source": str(source.resolve()),
                "source_sha256": sha256(source), "count": len(tasks), "files": files,
                "document_order": "source order; TriviaQA EntityPages then SearchResults",
                "oracle_support_filter": False, "paper_exact_split": False}
    write_json(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark", choices=["hotpotqa", "2wiki", "musique", "triviaqa", "canonical"], required=True)
    parser.add_argument("--split", choices=["train", "validation", "test"], required=True)
    parser.add_argument("--evidence-root", type=Path)
    parser.add_argument("--parquet", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare(**vars(args))))


if __name__ == "__main__":
    main()
