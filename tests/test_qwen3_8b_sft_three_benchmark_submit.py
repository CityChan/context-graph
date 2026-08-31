from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUBMITTER = ROOT / "scripts" / "submit_eval_qwen3_8b_sft_three_benchmarks.sh"
GAIA_RUNNER = ROOT / "scripts" / "train_bc_ctxgraph_8b_paperfaithful_5node_48h.sh"
DISCOVERY_RUNNER = ROOT / "scripts" / "eval_sab_react_30b_instruct_8node_smoke.sh"


def test_submitter_uses_the_merged_sft_model_and_cxtgraph_only():
    text = SUBMITTER.read_text(encoding="utf-8")
    assert "contextgraph_sft_models/954050_qwen3_8b_contextgraph_sft_32k_fullparam" in text
    assert "CONDA_ENV_NAME=cxtgraph" in text
    assert "deepseek_v4" not in text
    assert "DRY_RUN=${DRY_RUN:-0}" in text


def test_submitter_matches_each_controller_protocol():
    text = SUBMITTER.read_text(encoding="utf-8")
    assert "BC_CTXGRAPH_PROTOCOL=controller" in text
    assert "BC_CONTROLLER_ACTION_POLICY=structural" in text
    assert "BC_CONTEXT_LENGTH=65536" in text
    assert "BC_VAL_MAX_SAMPLES=-1" in text
    assert "BC_CONTROLLER_ACTION_POLICY=balanced" in text
    assert "VAL_DATA_FILE=data/gaia_validation_graph.parquet" in text
    assert "CONTEXT_LENGTH=32768" in text
    assert "SAB_CTXGRAPH_PROTOCOL=controller" in text
    assert "SAB_CONTROLLER_ACTION_POLICY=structural" in text
    assert "DISCOVERYBENCH_VAL_MAX_SAMPLES=239" in text


def test_submitter_requests_original_benchmark_topologies_and_disables_wandb():
    text = SUBMITTER.read_text(encoding="utf-8")
    assert "submit_job browsecomp sbatch" in text and "--nodes=4 --time=02:00:00" in text
    assert "submit_job gaia sbatch" in text and "--nodes=5 --time=03:00:00" in text
    assert "submit_job discoverybench sbatch" in text and "--nodes=4 --time=08:00:00" in text
    assert text.count("BC_DISABLE_WANDB=1") == 2
    assert "SAB_DISABLE_WANDB=1" in text


def test_submitter_uses_distinct_job_names_and_logs():
    text = SUBMITTER.read_text(encoding="utf-8")
    for job_name in (
        "eval-bc-sft8b-ctxgraph-controller-64k",
        "gaia-ctxgraph-controller-sft8b-32k",
        "eval-db-ctxgraph-controller-sft8b-4n",
    ):
        assert f"--job-name={job_name}" in text
    assert text.count("--output=") == 3
    assert text.count("--error=") == 3


def test_gaia_and_discovery_runners_accept_local_hf_models():
    for runner in (GAIA_RUNNER, DISCOVERY_RUNNER):
        text = runner.read_text(encoding="utf-8")
        assert 'if [ -d "$MODEL_PATH" ]' in text
        assert 'TRAINER_CACHE_DIR="$MODEL_PATH"' in text
        assert "local MODEL_PATH has no non-empty Hugging Face weight files" in text
    assert 'if [ "${BC_DISABLE_WANDB:-0}" = "1" ]' in GAIA_RUNNER.read_text(encoding="utf-8")
