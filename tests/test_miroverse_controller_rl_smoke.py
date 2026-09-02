from pathlib import Path

import pytest

from scripts.split_miroverse_rl_smoke_seeds import split_rows


ROOT = Path(__file__).resolve().parents[1]


def seed(query_hash: str) -> dict:
    return {
        "prompt": [{"role": "user", "content": query_hash}],
        "ability": "LocalSearch",
        "data_source": "miroverse_musique",
        "extra_info": {"query_hash": query_hash},
        "reward_model": {"style": "rule", "ground_truth": query_hash},
    }


def test_smoke_split_is_deterministic_and_query_disjoint():
    rows = [seed(str(index)) for index in range(12)]
    train, validation = split_rows(rows, train_samples=5, validation_samples=3, seed=42)
    repeated = split_rows(rows, train_samples=5, validation_samples=3, seed=42)
    train_hashes = {row["extra_info"]["query_hash"] for row in train}
    validation_hashes = {row["extra_info"]["query_hash"] for row in validation}
    assert (train, validation) == repeated
    assert len(train) == 5
    assert len(validation) == 3
    assert train_hashes.isdisjoint(validation_hashes)


def test_smoke_split_rejects_insufficient_unique_queries():
    with pytest.raises(ValueError, match="need 4 unique queries"):
        split_rows([seed("same"), seed("same")], train_samples=2, validation_samples=2, seed=42)


def test_miroverse_controller_rl_wrapper_uses_base_model_new_protocol_and_local_corpus():
    source = (ROOT / "scripts/smoke_train_miroverse_ctxgraph_controller_rl_qwen3_8b_5node.sh").read_text(encoding="utf-8")
    assert "#SBATCH -N 5" in source
    assert "MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-8B}" in source
    assert "BC_CTXGRAPH_PROTOCOL=controller" in source
    assert "BC_CONTROLLER_ACTION_POLICY=structural" in source
    assert "LOCAL_SEARCH_CORPUS=" in source
    assert "LOCAL_SEARCH_EMBEDDINGS=" in source
    assert "VAL_BEFORE_TRAIN=True" in source
    assert "TEST_FREQ=1" in source
    assert "TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-5}" in source


def test_contextgraph_runner_supports_absolute_data_and_local_retrieval_artifacts():
    source = (ROOT / "scripts/train_bc_ctxgraph_8b_paperfaithful_5node_48h.sh").read_text(encoding="utf-8")
    assert '/*) TRAIN_PARQUET="$TRAIN_DATA_FILE"' in source
    assert '/*) VAL_PARQUET="$VAL_DATA_FILE"' in source
    assert 'SEARCH_SERVER_ARGS+=(--local-corpus "$LOCAL_SEARCH_CORPUS" --local-embeddings "$LOCAL_SEARCH_EMBEDDINGS")' in source
    assert "WANDB_API_KEY=wandb_" not in source


def test_local_search_uses_configured_judge_model():
    source = (ROOT / "envs/local_search.py").read_text(encoding="utf-8")
    assert 'judge_model = os.getenv("JUDGE_MODEL", "gpt-5-nano")' in source
    assert "call_openai_raw(messages, model=judge_model)" in source
