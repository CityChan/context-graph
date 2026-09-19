from pathlib import Path

from scripts.summarize_alfworld_eval import extract_metrics


ROOT = Path(__file__).resolve().parents[1]
CONTROLLER_SFT_1NODE = ROOT / "scripts/eval_alfworld_ctxgraph_qwen3_8b_controller_sft_1node_idev.sh"
CONTROLLER_SFT_FULL = ROOT / "scripts/eval_alfworld_ctxgraph_qwen3_8b_controller_sft_1node_full.sh"


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


def test_corrected_contextgraph_launcher_uses_bounded_memory_without_checkpoints():
    launcher = (ROOT / "scripts/eval_alfworld_ctxgraph_8b_4node_zeroshot_corrected_idev.sh").read_text(encoding="utf-8")
    assert "export MODEL_PATH=Qwen/Qwen3-8B" in launcher
    assert "export ALFWORLD_VAL_ONLY=True" in launcher
    assert "export ALFWORLD_ROLLOUT_N=1" in launcher
    assert "export ALFWORLD_STRUCTURED_GRAPH_CONTROLLER=True" in launcher
    assert "export ALFWORLD_CONTROLLER_ACTION_POLICY=balanced" in launcher
    assert "export ALFWORLD_CONTROLLER_ALLOW_PASS=True" in launcher
    assert "export ALFWORLD_CONSOLIDATION_INTERVAL=0" in launcher
    assert "export ALFWORLD_ENABLE_RETRIEVAL_MEMORY=True" in launcher
    assert "export ALFWORLD_INJECT_GRAPH_STATE_AFTER_ACTION=False" in launcher
    assert "export ALFWORLD_TRAINER_RESUME_MODE=disable" in launcher
    assert "ALFWORLD_EVAL_SAMPLES=${ALFWORLD_EVAL_SAMPLES:-32}" in launcher
    assert "ALFWORLD_DATA_SEED=${ALFWORLD_DATA_SEED:-42}" in launcher


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
    assert "export ALFWORLD_CONTROLLER_ACTION_POLICY=balanced" in launcher
    assert "export ALFWORLD_CONSOLIDATION_INTERVAL=0" in launcher
    assert "export ALFWORLD_TRAINER_RESUME_MODE=disable" in launcher


def test_foldagent_zeroshot_launcher_matches_contextgraph_eval_budget():
    launcher = (ROOT / "scripts/eval_alfworld_foldagent_8b_4node_zeroshot_idev.sh").read_text(encoding="utf-8")
    assert "export MODEL_PATH=Qwen/Qwen3-8B" in launcher
    assert "export ALFWORLD_AGENT_LOOP=fold_agent" in launcher
    assert "export ALFWORLD_WORKFLOW=alfworld_branch" in launcher
    assert "export ALFWORLD_PROCESS_REWARD='[flat,scope]'" in launcher
    assert "export ALFWORLD_DATA_VARIANT=alfworld_branch" in launcher
    assert "ALFWORLD_EVAL_SAMPLES=${ALFWORLD_EVAL_SAMPLES:-32}" in launcher
    assert "ALFWORLD_DATA_SEED=${ALFWORLD_DATA_SEED:-42}" in launcher
    assert "export ALFWORLD_PROMPT_LENGTH=4096" in launcher
    assert "export ALFWORLD_RESPONSE_LENGTH=12288" in launcher
    assert "export ALFWORLD_MAX_TOKEN_LEN_PER_GPU=16384" in launcher
    assert "export ALFWORLD_VAL_MAX_TURN=60" in launcher
    assert "export ALFWORLD_VAL_MAX_SESSION=3" in launcher
    assert "export ALFWORLD_STRUCTURED_GRAPH_CONTROLLER=False" in launcher
    assert "export ALFWORLD_CONTROLLER_ACTION_POLICY=balanced" in launcher
    assert "export ALFWORLD_CONSOLIDATION_INTERVAL=0" in launcher
    assert "export ALFWORLD_TRAINER_RESUME_MODE=disable" in launcher


def test_shared_alfworld_launcher_wires_controller_and_resume_mode():
    launcher = (ROOT / "scripts/train_alfworld_ctxgraph_8b_4node_30step.sh").read_text(encoding="utf-8")
    assert "plugin.structured_graph_controller=${ALFWORLD_STRUCTURED_GRAPH_CONTROLLER}" in launcher
    assert "plugin.controller_owned_tool_formatting=${ALFWORLD_CONTROLLER_OWNED_TOOL_FORMATTING}" in launcher
    assert "plugin.controller_action_policy=${ALFWORLD_CONTROLLER_ACTION_POLICY}" in launcher
    assert "plugin.controller_allow_pass=${ALFWORLD_CONTROLLER_ALLOW_PASS}" in launcher
    assert "plugin.enable_retrieval_memory=${ALFWORLD_ENABLE_RETRIEVAL_MEMORY}" in launcher
    assert "plugin.inject_graph_state_after_action=${ALFWORLD_INJECT_GRAPH_STATE_AFTER_ACTION}" in launcher
    assert "plugin.consolidation_interval=${ALFWORLD_CONSOLIDATION_INTERVAL}" in launcher
    assert "trainer.resume_mode=${ALFWORLD_TRAINER_RESUME_MODE}" in launcher
    assert "export TORCH_COMPILE_DISABLE=${TORCH_COMPILE_DISABLE:-1}" in launcher


def test_single_node_controller_sft_eval_uses_heldout_alfworld_and_structural_policy():
    launcher = CONTROLLER_SFT_1NODE.read_text(encoding="utf-8")
    assert "998826_miroverse_qwen3_8b_lora32_4k_merged" in launcher
    assert 'if [ -z "${SLURM_JOB_NODELIST:-}" ]' in launcher
    assert "MAX_SAMPLES=${MAX_SAMPLES:-8}" in launcher
    assert "data/alfworld_graph_real_test.parquet" in launcher
    assert "--guided-decoding-backend guidance" in launcher
    assert "--reasoning-parser qwen3" in launcher
    assert "export TORCH_COMPILE_DISABLE=${TORCH_COMPILE_DISABLE:-1}" in launcher
    assert "--structured-graph-controller" in launcher
    assert "--controller-action-policy structural" in launcher
    assert "--no-controller-allow-pass" in launcher
    assert "--consolidation-interval" in launcher
    assert "--reasoning-effort non-thinking" in launcher
    assert "scripts/validate_contextgraph_traces.py" in launcher


def test_single_node_controller_sft_eval_supports_full_split():
    launcher = CONTROLLER_SFT_1NODE.read_text(encoding="utf-8")
    assert 'if [ "$MAX_SAMPLES" -eq 0 ]' in launcher
    assert "DATA_N_VAL=0" in launcher
    assert "EPISODE_LABEL=all" in launcher
    assert '--n_val "$DATA_N_VAL"' in launcher
    assert '--max-samples "$MAX_SAMPLES"' in launcher
    assert '--num-workers "$NUM_WORKERS"' in launcher


def test_controller_sft_full_sbatch_runs_all_supported_alfworld_tasks():
    launcher = CONTROLLER_SFT_FULL.read_text(encoding="utf-8")
    assert "#SBATCH -p gh" in launcher
    assert "#SBATCH -N 1" in launcher
    assert "#SBATCH -t 12:00:00" in launcher
    assert "#SBATCH -A AST24021" in launcher
    assert "export MAX_SAMPLES=${MAX_SAMPLES:-0}" in launcher
    assert "export START_INDEX=${START_INDEX:-0}" in launcher
    assert "export DATA_SEED=${DATA_SEED:-42}" in launcher
    assert "export NUM_WORKERS=${NUM_WORKERS:-4}" in launcher
    assert "eval_alfworld_ctxgraph_qwen3_8b_controller_sft_1node_idev.sh" in launcher


def test_api_evaluator_wires_controller_policy_controls():
    evaluator = (ROOT / "scripts/eval_interactive.py").read_text(encoding="utf-8")
    agent_utils = (ROOT / "agents/utils.py").read_text(encoding="utf-8")
    assert '"algorithm": {"adv_estimator": "foldgrpo"}' in evaluator
    assert 'chat_template_kwargs["enable_thinking"] = False' in evaluator
    assert 'chat_template_kwargs["enable_thinking"] = False' in agent_utils
    assert '"GRAMMAR_ACTIVE"' in evaluator
    assert '"Reply exactly NOT_JSON with no braces."' in evaluator
    assert '["response_format", "structured_outputs", "guided_json"]' in evaluator
    assert '"api_structured_output_mode": args.api_structured_output_mode' in evaluator
    assert '"--controller-action-policy"' in evaluator
    assert '"--controller-allow-pass"' in evaluator
    assert '"--inject-graph-state-after-action"' in evaluator
    assert '"controller_action_policy": args.controller_action_policy' in evaluator
    assert '"controller_allow_pass": args.controller_allow_pass' in evaluator
    assert '"inject_graph_state_after_action": args.inject_graph_state_after_action' in evaluator


def test_sbatch_launcher_requests_four_gh_nodes_and_delegates_to_eval():
    launcher = (ROOT / "scripts/sbatch_alfworld_ctxgraph_8b_4node_zeroshot.sh").read_text(encoding="utf-8")
    assert "#SBATCH -p gh" in launcher
    assert "#SBATCH -N 4" in launcher
    assert "#SBATCH -t 04:00:00" in launcher
    assert "#SBATCH -A AST24021" in launcher
    assert "exec bash scripts/eval_alfworld_ctxgraph_8b_4node_zeroshot_idev.sh" in launcher
