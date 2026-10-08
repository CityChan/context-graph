"""Cache and probe the pinned CPU A-MEM embedder; never install into cxtgraph."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    from huggingface_hub import snapshot_download
    from agents.graph_memory_baselines import EMBEDDING_MODEL, EMBEDDING_REVISION, embed, AMemMemory
    # Fail before model servers start if the existing agent environment lacks ST.
    import sentence_transformers
    path = snapshot_download(EMBEDDING_MODEL, revision=EMBEDDING_REVISION,
        local_files_only=args.offline,
        allow_patterns=["*.json", "*.txt", "*.safetensors"])
    vector = embed("Research evidence with source docids.")
    store = AMemMemory()
    metadata = dict(keywords=[], context="probe", tags=[])
    first = store.add("Paris is the capital of France.", metadata)
    store.add("Water freezes at zero degrees Celsius.", metadata)
    if store.nearest("Paris is the capital of France.", 1) != [first]:
        raise RuntimeError("A-MEM semantic retrieval probe failed")
    print(json.dumps(dict(event="AMEM_EMBEDDING_OK", model=EMBEDDING_MODEL,
        revision=EMBEDDING_REVISION, path=path, dimension=len(vector),
        semantic_retrieval_passed=True,
        sentence_transformers=sentence_transformers.__version__)))


if __name__ == "__main__":
    main()
