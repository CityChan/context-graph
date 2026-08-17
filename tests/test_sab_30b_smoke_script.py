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
    assert "SAB_DATA_SEED=42" in source


def test_30b_runner_supports_all_three_matched_methods():
    source = Path("scripts/eval_sab_react_30b_instruct_8node_smoke.sh").read_text(encoding="utf-8")

    expected = {
        "react": ("react_agent_code", "code", "[flat]", "data/sab_test_code.parquet"),
        "fold": ("fold_agent_code", "code_branch", "[flat,scope]", "data/sab_test_code_branch.parquet"),
        "ctxgraph": (
            "context_graph_code_isolated_agent",
            "code_graph",
            "[flat,scope,graph]",
            "data/sab_test_code_graph.parquet",
        ),
    }
    for method, values in expected.items():
        assert f"  {method})" in source
        for value in values:
            assert value in source

    assert "data.seed=$SAB_DATA_SEED" in source
    assert "plugin.max_session=$SAB_MAX_SESSION" in source
    assert "plugin.branch_len=$SAB_BRANCH_LEN" in source


def test_formal_suite_submits_react_fold_and_contextgraph():
    source = Path("scripts/submit_sab_30b_instruct_8node_formal_suite.sh").read_text(encoding="utf-8")

    assert "for method in react fold ctxgraph" in source
    assert 'SAB_METHOD="$method"' in source
    assert "submit_sab_react_30b_instruct_8node_formal.sh" in source
