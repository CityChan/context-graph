from pathlib import Path

from scripts.prepare_gsm8k_grpo_data import convert_example, extract_ground_truth


def test_extract_ground_truth_normalizes_commas():
    assert extract_ground_truth("work\n#### 1,234") == "1234"


def test_convert_example_uses_verl_gsm8k_schema():
    row = convert_example(
        {"question": "What is 6 times 7?", "answer": "6 * 7 = 42\n#### 42"},
        "train",
        3,
    )

    assert row["data_source"] == "openai/gsm8k"
    assert row["prompt"][0]["role"] == "user"
    assert 'after "####"' in row["prompt"][0]["content"]
    assert row["reward_model"] == {"style": "rule", "ground_truth": "42"}
    assert row["extra_info"] == {"split": "train", "index": 3}


def test_launcher_carries_vista_cuda_and_disables_vllm_sleep_mode():
    source = Path("scripts/smoke_train_gsm8k_grpo_1node_10step.sh").read_text()

    assert "CUDA_TARGET_LIB=$CUDA_HOME/targets/sbsa-linux/lib" in source
    assert 'LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:$CUDA_TARGET_LIB:$CUDA_LIB:' in source
    assert "actor_rollout_ref.rollout.free_cache_engine=False" in source
    assert "+actor_rollout_ref.rollout.engine_kwargs.vllm.enable_sleep_mode=False" in source


def test_launcher_isolates_jit_caches_and_uses_eager_mode():
    source = Path("scripts/smoke_train_gsm8k_grpo_1node_10step.sh").read_text()

    assert "VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-/tmp/contextgraph-gsm8k-vllm-$CACHE_TAG}" in source
    assert "TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/tmp/contextgraph-gsm8k-inductor-$CACHE_TAG}" in source
    assert "TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/contextgraph-gsm8k-triton-$CACHE_TAG}" in source
    assert "TORCHDYNAMO_DISABLE=1" in source
    assert "actor_rollout_ref.rollout.enforce_eager=True" in source
    assert source.count("fsdp_config.use_torch_compile=False") == 2


def test_launcher_matches_proven_vista_runtime_preamble():
    source = Path("scripts/smoke_train_gsm8k_grpo_1node_10step.sh").read_text()

    required_settings = [
        'export PATH="${CONDA_PREFIX}/bin:$CUDA_HOME/bin:$PATH"',
        'export CPATH="$NVPL_INCLUDE:$CUDA_INCLUDE:${CPATH:-}"',
        'export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"',
        "export CC=${GSM8K_CC:-gcc}",
        "export CXX=${GSM8K_CXX:-g++}",
        "export CUDAHOSTCXX=${GSM8K_CUDAHOSTCXX:-g++}",
        "export FLASHINFER_WORKSPACE_BASE=/tmp",
        "export HF_HUB_DISABLE_FILE_LOCKING=1",
        "export VLLM_WORKER_MULTIPROC_METHOD=spawn NCCL_P2P_LEVEL=NVL",
        "export RAY_memory_usage_threshold=0.99 RAY_memory_monitor_refresh_ms=0",
    ]
    for setting in required_settings:
        assert setting in source


def test_launcher_preflights_runtime_and_cleans_up_ray():
    source = Path("scripts/smoke_train_gsm8k_grpo_1node_10step.sh").read_text()

    assert "ctypes.CDLL('libnvrtc.so.12')" in source
    assert "torch.cuda.is_available()" in source
    assert "import vllm, verl" in source
    assert "trap cleanup EXIT INT TERM" in source
    assert "actor_rollout_ref.rollout.max_num_seqs=64" in source
