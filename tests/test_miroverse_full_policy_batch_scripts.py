from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BATCH = ROOT / "scripts" / "generate_miroverse_full_policy_deepseek_v4_4node_batch.sh"
SUBMIT = ROOT / "scripts" / "submit_miroverse_full_policy_deepseek_v4_4node_shards.sh"
MERGE = ROOT / "scripts" / "merge_miroverse_full_policy_shards.sh"


def test_batch_uses_verified_four_node_concurrency_and_completion_marker():
    text = BATCH.read_text(encoding="utf-8")
    assert "#SBATCH -N 4" in text
    assert "export NUM_WORKERS=${NUM_WORKERS:-2}" in text
    assert "export MAX_NUM_SEQS=${MAX_NUM_SEQS:-2}" in text
    assert "export COLOCATE_SEARCH=1" in text
    assert 'touch "$ARTIFACT_ROOT/.complete"' in text


def test_submitter_serializes_shards_for_four_node_quota():
    text = SUBMIT.read_text(encoding="utf-8")
    assert "SHARD_SIZE=${SHARD_SIZE:-1500}" in text
    assert '--dependency="afterok:$previous_job"' in text
    assert "submission_plan.tsv" in text
    assert "--parsable" in text


def test_merger_only_reads_complete_shards_and_recurates_raw_results():
    text = MERGE.read_text(encoding="utf-8")
    assert "submission_plan.tsv" in text
    assert 'if [ ! -f "$shard_dir/.complete" ]' in text
    assert "gaia_results_*.json" in text
    assert "build_contextgraph_full_policy_sft.py" in text
    assert "contextgraph_sft_validation.parquet" in text


def test_full_trace_token_telemetry_suppresses_model_length_warning():
    text = (ROOT / "agents" / "graph_agent_isolated.py").read_text(encoding="utf-8")
    assert "truncation=False" in text
    assert "verbose=False" in text
