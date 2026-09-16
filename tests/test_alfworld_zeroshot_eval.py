from pathlib import Path

from scripts.summarize_alfworld_eval import extract_metrics


ROOT = Path(__file__).resolve().parents[1]


def test_extract_metrics_from_verl_step_line():
    text = (
        "(TaskRunner pid=1) step:0 - val/reward:0.2575 - "
        "val/task_reward:np.float64(0.25) - val/finish_rate:0.25 - "
        "val/no_finish_rate:0.75 - val/structured_graph_controller:1.0"
    )
    assert extract_metrics(text) == {
        "reward": 0.2575,
        "task_reward": 0.25,
        "finish_rate": 0.25,
        "no_finish_rate": 0.75,
        "structured_graph_controller": 1.0,
    }


def test_zeroshot_launcher_uses_fixed_base_model_and_current_controller():
    launcher = (ROOT / "scripts/eval_alfworld_ctxgraph_8b_4node_zeroshot_idev.sh").read_text(encoding="utf-8")
    assert "export MODEL_PATH=Qwen/Qwen3-8B" in launcher
    assert "export ALFWORLD_VAL_ONLY=True" in launcher
    assert "export ALFWORLD_ROLLOUT_N=1" in launcher
    assert "export ALFWORLD_STRUCTURED_GRAPH_CONTROLLER=True" in launcher
    assert "export ALFWORLD_CONTROLLER_ACTION_POLICY=structural" in launcher
    assert "export ALFWORLD_TRAINER_RESUME_MODE=disable" in launcher
    assert "ALFWORLD_EVAL_SAMPLES=${ALFWORLD_EVAL_SAMPLES:-32}" in launcher


def test_react_zeroshot_launcher_matches_contextgraph_eval_budget():
    launcher = (ROOT / "scripts/eval_alfworld_react_8b_4node_zeroshot_idev.sh").read_text(encoding="utf-8")
    assert "export MODEL_PATH=Qwen/Qwen3-8B" in launcher
    assert "export ALFWORLD_AGENT_LOOP=react_agent" in launcher
    assert "export ALFWORLD_WORKFLOW=alfworld" in launcher
    assert "export ALFWORLD_PROCESS_REWARD='[flat]'" in launcher
    assert "export ALFWORLD_DATA_VARIANT=alfworld" in launcher
    assert "ALFWORLD_EVAL_SAMPLES=${ALFWORLD_EVAL_SAMPLES:-32}" in launcher
    assert "ALFWORLD_DATA_SEED=${ALFWORLD_DATA_SEED:-42}" in launcher
    assert "export ALFWORLD_PROMPT_LENGTH=4096" in launcher
    assert "export ALFWORLD_RESPONSE_LENGTH=12288" in launcher
    assert "export ALFWORLD_MAX_TOKEN_LEN_PER_GPU=16384" in launcher
    assert "export ALFWORLD_VAL_MAX_TURN=60" in launcher
    assert "export ALFWORLD_STRUCTURED_GRAPH_CONTROLLER=False" in launcher
    assert "export ALFWORLD_CONSOLIDATION_INTERVAL=0" in launcher
    assert "export ALFWORLD_TRAINER_RESUME_MODE=disable" in launcher


def test_shared_alfworld_launcher_wires_controller_and_resume_mode():
    launcher = (ROOT / "scripts/train_alfworld_ctxgraph_8b_4node_30step.sh").read_text(encoding="utf-8")
    assert "plugin.structured_graph_controller=${ALFWORLD_STRUCTURED_GRAPH_CONTROLLER}" in launcher
    assert "plugin.controller_owned_tool_formatting=${ALFWORLD_CONTROLLER_OWNED_TOOL_FORMATTING}" in launcher
    assert "plugin.controller_action_policy=${ALFWORLD_CONTROLLER_ACTION_POLICY}" in launcher
    assert "plugin.consolidation_interval=${ALFWORLD_CONSOLIDATION_INTERVAL}" in launcher
    assert "trainer.resume_mode=${ALFWORLD_TRAINER_RESUME_MODE}" in launcher


def test_sbatch_launcher_requests_four_gh_nodes_and_delegates_to_eval():
    launcher = (ROOT / "scripts/sbatch_alfworld_ctxgraph_8b_4node_zeroshot.sh").read_text(encoding="utf-8")
    assert "#SBATCH -p gh" in launcher
    assert "#SBATCH -N 4" in launcher
    assert "#SBATCH -t 04:00:00" in launcher
    assert "#SBATCH -A AST24021" in launcher
    assert "exec bash scripts/eval_alfworld_ctxgraph_8b_4node_zeroshot_idev.sh" in launcher
