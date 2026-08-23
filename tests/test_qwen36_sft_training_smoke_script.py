from pathlib import Path


SCRIPT = Path("scripts/smoke_train_contextgraph_sft_qwen36_27b_4node_idev.sh")
CHECKER = Path("scripts/check_contextgraph_sft_data.py")


def test_qwen36_training_smoke_runs_a_real_multiturn_optimizer_step():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "Qwen/Qwen3.6-27B" in text
    assert "verl.trainer.fsdp_sft_trainer" in text
    assert "data.multiturn.enable=True" in text
    assert "trainer.total_training_steps=\"$TOTAL_TRAINING_STEPS\"" in text
    assert "trainer.save_freq=1" in text
    assert "contextgraph_sft_train.parquet" in text


def test_qwen36_training_smoke_is_safe_for_one_row_on_four_nodes():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "EXPECTED_NUM_NODES=${EXPECTED_NUM_NODES:-4}" in text
    assert "ulysses_sequence_parallel_size=\"$NUM_NODES\"" in text
    assert "use_remove_padding=True" in text
    assert "TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1}" in text
    assert "LORA_RANK=${LORA_RANK:-32}" in text
    assert "model.strategy=fsdp2" in text


def test_qwen36_training_smoke_checks_every_node_and_checkpoint_shard():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "SFT_PREFLIGHT_WORKER=1" in text
    assert "memory.used,memory.free" in text
    assert "model_world_size_*_rank_*.pt" in text
    assert 'if [ "$MODEL_SHARDS" -ne "$NUM_NODES" ]' in text
    assert "check_contextgraph_sft_data.py" in text


def test_qwen36_training_smoke_ignores_inherited_shared_hf_cache():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "HF_HOME=${SFT_HF_HOME:-$SCRATCH/hf_cache}" in text
    assert "HF_HUB_CACHE=${SFT_HF_HUB_CACHE:-$HF_HOME/hub}" in text
    assert 'require_scratch_path HF_HUB_CACHE "$HF_HUB_CACHE"' in text


def test_contextgraph_sft_checker_requires_loss_tokens():
    text = CHECKER.read_text(encoding="utf-8")
    assert 'required_columns = {"messages", "tools", "enable_thinking"}' in text
    assert 'sample["loss_mask"]' in text
    assert "tokenized SFT sample has no assistant loss tokens" in text
