from pathlib import Path

import pandas as pd
import pytest

from scripts.audit_skillrl_search_reference import extract_last_number, summarize_log
from scripts.sample_searchr1_reference_data import stratified_sample


def test_stratified_sample_is_balanced_and_deterministic():
    frame = pd.DataFrame(
        {
            "data_source": ["searchR1_nq"] * 5 + ["searchR1_hotpotqa"] * 5,
            "row_id": list(range(10)),
        }
    )
    sources = ("searchR1_nq", "searchR1_hotpotqa")
    first = stratified_sample(frame, sources, per_source=3, seed=7)
    second = stratified_sample(frame, sources, per_source=3, seed=7)

    assert first.equals(second)
    assert first["data_source"].value_counts().to_dict() == {
        "searchR1_nq": 3,
        "searchR1_hotpotqa": 3,
    }


def test_stratified_sample_rejects_short_source():
    frame = pd.DataFrame({"data_source": ["searchR1_nq"]})
    with pytest.raises(ValueError, match="fewer than requested"):
        stratified_sample(frame, ("searchR1_nq",), per_source=2, seed=0)


def test_audit_parser_handles_numpy_wrapped_and_plain_metrics(tmp_path: Path):
    log = tmp_path / "run.log"
    log.write_text(
        "{'val/searchR1_nq/test_score': np.float64(0.25), "
        "'val/searchR1_hotpotqa/test_score': 0.5, 'actor/pg_loss': -1.25e-2}\n",
        encoding="utf-8",
    )

    scores, health, score_history = summarize_log(log)
    assert scores == {"searchR1_nq": 0.25, "searchR1_hotpotqa": 0.5}
    assert score_history == {"searchR1_nq": [0.25], "searchR1_hotpotqa": [0.5]}
    assert health["actor/pg_loss"] == pytest.approx(-0.0125)
    assert extract_last_number("actor/grad_norm: 1.5", "actor/grad_norm") == 1.5


def test_audit_parser_uses_last_metric_value(tmp_path: Path):
    log = tmp_path / "run.log"
    log.write_text(
        "actor/grad_norm: 1.5\nactor/grad_norm: 0.75\nactor/kl_loss: 2e-4\n",
        encoding="utf-8",
    )

    _, health, _ = summarize_log(log)
    assert health["actor/grad_norm"] == 0.75
    assert health["actor/kl_loss"] == pytest.approx(0.0002)
