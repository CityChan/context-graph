from pathlib import Path


SCRIPT = Path("scripts/smoke_train_contextgraph_sft_qwen36_27b_4node_idev.sh")
CHECKER = Path("scripts/check_contextgraph_sft_data.py")
MULTITURN_DATASET = Path("verl/utils/dataset/multiturn_sft_dataset.py")


def test_qwen36_training_smoke_runs_a_real_multiturn_optimizer_step():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "Qwen/Qwen3.6-27B" in text
    assert "verl.trainer.fsdp_sft_trainer" in text
    assert "data.multiturn.enable=True" in text
    assert "data.multiturn.loss_mask_mode=assistant_tokens" in text
    assert "trainer.total_training_steps=\"$TOTAL_TRAINING_STEPS\"" in text
    assert "trainer.save_freq=1" in text
    assert "contextgraph_sft_train.parquet" in text


def test_qwen36_training_smoke_is_safe_for_one_row_on_four_nodes():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "EXPECTED_NUM_NODES=${EXPECTED_NUM_NODES:-4}" in text
    assert "ulysses_sequence_parallel_size=1" in text
    assert "use_remove_padding=False" in text
    assert "TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-4}" in text
    assert 'TRAIN_FILES="[$TRAIN_FILE,$TRAIN_FILE,$TRAIN_FILE,$TRAIN_FILE]"' in text
    assert "LORA_RANK=${LORA_RANK:-32}" in text
    assert "model.strategy=fsdp2" in text
    assert "ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-sdpa}" in text
    assert 'model.attn_implementation="$ATTN_IMPLEMENTATION"' in text


def test_qwen36_training_smoke_checks_every_node_and_checkpoint_shard():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "SFT_PREFLIGHT_WORKER=1" in text
    assert "memory.used,memory.free" in text
    assert "model_world_size_*_rank_*.pt" in text
    assert 'if [ "$MODEL_SHARDS" -ne "$NUM_NODES" ]' in text
    assert "check_contextgraph_sft_data.py" in text
    assert "DATA_PREFLIGHT_TIMEOUT=${DATA_PREFLIGHT_TIMEOUT:-300}" in text
    assert 'timeout --foreground "${DATA_PREFLIGHT_TIMEOUT}s" python -u' in text


def test_qwen36_training_smoke_ignores_inherited_shared_hf_cache():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "HF_HOME=${SFT_HF_HOME:-$SCRATCH/hf_cache}" in text
    assert "HF_HUB_CACHE=${SFT_HF_HUB_CACHE:-$HF_HOME/hub}" in text
    assert 'require_scratch_path HF_HUB_CACHE "$HF_HUB_CACHE"' in text


def test_qwen36_training_smoke_uses_deepseek_env_without_external_flash_attn():
    text = SCRIPT.read_text(encoding="utf-8")
    requirements = Path("requirements_qwen36_sft.txt").read_text(encoding="utf-8")
    assert "TRAIN_CONDA_ENV=${TRAIN_CONDA_ENV:-deepseek_v4}" in text
    assert "import accelerate, codetiming, datasets, hydra" in text
    assert "flash_attn" not in text
    assert "pandas==2.3.3" in requirements
    assert "datasets==4.4.1" in requirements
    assert "qwen-vl-utils==0.0.14" in requirements
    assert "peft==0.18.1" in requirements
    assert "torchdata==0.11.0" in requirements
    assert "import verl.trainer.fsdp_sft_trainer" in text


def test_contextgraph_sft_checker_requires_loss_tokens():
    text = CHECKER.read_text(encoding="utf-8")
    assert 'required_columns = {"messages", "tools", "enable_thinking"}' in text
    assert 'sample["loss_mask"]' in text
    assert "tokenized SFT sample has no assistant loss tokens" in text
    assert 'stage("load tokenizer")' in text
    assert 'stage("tokenize first sample")' in text
    assert "local_files_only=True" in text
    assert '"loss_mask_mode": "assistant_tokens"' in text


def test_multiturn_sft_uses_renderable_conversation_prefixes():
    text = MULTITURN_DATASET.read_text(encoding="utf-8")
    assert "conversation_prefix=messages[: i + 1]" in text
    assert 'turn.get("role") == "user"' in text
    assert "previous_inputs = render_prefix(conversation_prefix[:-1])" in text
    assert "Chat template tokenization is not prefix-stable" in text
    assert "if len(prefix) == 0:" in text
    assert "messages: list = convert_nested_value_to_list_recursive(example[self.messages_key])" in text
    assert 'self.loss_mask_mode in ["per_message", "assistant_tokens"]' in text
    assert "return_assistant_tokens_mask=True" in text
    assert 'if self.loss_mask_mode == "assistant_tokens":' in text
