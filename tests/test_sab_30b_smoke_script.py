from pathlib import Path


def test_real_sab_eval_loads_and_validates_visual_judge_credentials():
    source = Path("scripts/eval_sab_react_30b_instruct_8node_smoke.sh").read_text(encoding="utf-8")
    ray_start = source.index("ray start --head")

    assert "$WORK/.openai_env" in source[:ray_start]
    assert "OPENAI_API_KEY" in source[:ray_start]
    assert "AZURE_OPENAI_KEY" in source[:ray_start]
    assert "AZURE_OPENAI_DEPLOYMENT_NAME" in source[:ray_start]
    assert "gpt4_visual_judge.py" in source[:ray_start]
