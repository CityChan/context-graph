from pathlib import Path


def test_real_sab_eval_loads_and_validates_visual_judge_credentials():
    source = Path("scripts/eval_sab_react_30b_instruct_8node_smoke.sh").read_text(encoding="utf-8")
    ray_start = source.index("ray start --head")

    assert "$WORK/.openai_env" in source[:ray_start]
    assert "OPENAI_API_KEY" in source[:ray_start]
    assert "AZURE_OPENAI_KEY" in source[:ray_start]
    assert "AZURE_OPENAI_DEPLOYMENT_NAME" in source[:ray_start]
    assert "gpt4_visual_judge.py" in source[:ray_start]


def test_formal_submit_uses_full_matched_sab_budget():
    source = Path("scripts/submit_sab_react_30b_instruct_8node_formal.sh").read_text(encoding="utf-8")

    assert "SAB_RUN_TAG=formal" in source
    assert "SAB_REAL_EVAL=1" in source
    assert "SAB_DUMP_VALIDATION=1" in source
    assert "SAB_VAL_MAX_SAMPLES=-1" in source
    assert "SAB_PROMPT_LENGTH=16384" in source
    assert "SAB_RESPONSE_LENGTH=24576" in source
    assert "SAB_MAX_TOKEN_LEN_PER_GPU=40960" in source
    assert "SAB_VAL_MAX_TURN=32" in source
    assert "SAB_TURN_MAX_NEW_TOKENS=2048" in source
