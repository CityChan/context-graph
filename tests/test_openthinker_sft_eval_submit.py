from pathlib import Path


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_openthinker_submit_wires_model_to_bc_and_gaia():
    source = _read("scripts/submit_eval_openthinker_sft_8b_bc_gaia.sh")

    assert "open-thoughts/OpenThinkerAgent-8B-ColdStartSFTForRL" in source
    assert "scripts/submit_eval_bc_8b_4node_zeroshot_64k.sh" in source
    assert "scripts/submit_gaia_benchmark_8b_5node.sh" in source
    assert "scripts/check_hf_model_support.py" in source
    assert 'PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"' in source
    assert "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1" in source
    assert "BC_EXPERIMENT_MODEL_TAG=openthinker_sft_8b" in source
    assert "GAIA_EXPERIMENT_MODEL_TAG=openthinker_sft_8b" in source


def test_base_submitters_propagate_model_identity():
    bc_submit = _read("scripts/submit_eval_bc_8b_4node_zeroshot_64k.sh")
    bc_eval = _read("scripts/eval_bc_baseline_8b_4node_zeroshot.sh")
    gaia_submit = _read("scripts/submit_gaia_benchmark_8b_5node.sh")

    assert 'MODEL_PATH="$MODEL_PATH"' in bc_submit
    assert 'BC_EXPERIMENT_MODEL_TAG="$BC_EXPERIMENT_MODEL_TAG"' in bc_submit
    assert "BC_EXPERIMENT_MODEL_TAG=${BC_EXPERIMENT_MODEL_TAG:-8b}" in bc_eval
    assert "MODEL_PATH=$GAIA_MODEL_PATH" in gaia_submit
    assert "GAIA_EXPERIMENT_MODEL_TAG" in gaia_submit
