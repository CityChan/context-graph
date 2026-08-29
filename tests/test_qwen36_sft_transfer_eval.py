from pathlib import Path


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_model_merger_supports_adapter_only_export_with_explicit_alpha():
    source = _read("verl/model_merger/base_model_merger.py")
    assert '"--lora-adapter-only"' in source
    assert '"--lora-alpha"' in source
    assert "self.config.lora_alpha or 0" in source
    assert "Checkpoint contains no LoRA parameters" in source


def test_export_script_keeps_only_a_verified_lora_adapter():
    source = _read("scripts/export_contextgraph_sft_lora.sh")
    assert "--lora-adapter-only" in source
    assert '--lora-alpha "$LORA_ALPHA"' in source
    assert "adapter_model.safetensors" in source
    assert "EXPORT_ROOT already exists" in source


def test_all_three_eval_runners_accept_the_same_lora_overrides():
    runners = (
        "scripts/eval_bc_baseline_8b_4node_zeroshot.sh",
        "scripts/train_bc_ctxgraph_8b_paperfaithful_5node_48h.sh",
        "scripts/eval_sab_react_30b_instruct_8node_smoke.sh",
    )
    for path in runners:
        source = _read(path)
        assert 'LORA_ADAPTER_PATH=${LORA_ADAPTER_PATH:-}' in source
        assert 'actor_rollout_ref.model.lora_adapter_path="$LORA_ADAPTER_PATH"' in source
        assert 'actor_rollout_ref.model.lora_rank="$LORA_RANK"' in source
        assert 'actor_rollout_ref.model.lora_alpha="$LORA_ALPHA"' in source
        assert '"${MODEL_LORA_ARGS[@]}"' in source


def test_paired_transfer_submitter_is_matched_and_complete():
    source = _read("scripts/submit_eval_qwen36_27b_sft_transfer.sh")
    assert "for variant in base sft" in source
    assert "BENCHMARKS=${BENCHMARKS:-gaia,bc,discovery}" in source
    assert "GAIA_METHODS=ctxgraph" in source
    assert "BC_METHODS=contextgraph" in source
    assert "DISCOVERYBENCH_METHODS=ctxgraph" in source
    assert "GAIA_CONTROLLER_ACTION_POLICY=balanced" in source
    assert "BC_CONTROLLER_ACTION_POLICY=balanced" in source
    assert "DISCOVERYBENCH_CONTROLLER_ACTION_POLICY=balanced" in source
    assert "LORA_RANK=${LORA_RANK:-32}" in source
    assert "LORA_ALPHA=${LORA_ALPHA:-64}" in source
    assert "EVAL_MAX_SAMPLES=${EVAL_MAX_SAMPLES:--1}" in source
    assert "DEFAULT_EVAL_TIME=00:30:00" in source
    assert "DEFAULT_EVAL_TIME=01:30:00" in source
    assert "GAIA_EVAL_TIME=${GAIA_EVAL_TIME:-$DEFAULT_EVAL_TIME}" in source
    assert "BC_EVAL_TIME=${BC_EVAL_TIME:-$DEFAULT_EVAL_TIME}" in source
    assert (
        "DISCOVERYBENCH_TIME_LIMIT="
        "${DISCOVERYBENCH_TIME_LIMIT:-$DEFAULT_EVAL_TIME}"
    ) in source
    assert 'GAIA_EVAL_TIME="$GAIA_EVAL_TIME"' in source
    assert 'BC_EVAL_TIME="$BC_EVAL_TIME"' in source
    assert 'DISCOVERYBENCH_TIME_LIMIT="$DISCOVERYBENCH_TIME_LIMIT"' in source
    assert "DRY_RUN=${DRY_RUN:-0}" in source
    assert "TORCHDYNAMO_DISABLE=0" in source
    assert "VLLM_USE_AOT_COMPILE=1" in source


def test_idev_transfer_runner_uses_current_allocation_and_one_sample_default():
    source = _read("scripts/eval_qwen36_27b_sft_transfer_idev.sh")
    assert 'EVAL_MAX_SAMPLES=${EVAL_MAX_SAMPLES:-1}' in source
    assert 'scontrol show hostnames "$SLURM_JOB_NODELIST"' in source
    assert "sbatch" not in source
    assert "BENCHMARK=${BENCHMARK:-bc}" in source
    assert "VARIANT=${VARIANT:-base}" in source
    assert "BC_CTXGRAPH_PROTOCOL=controller" in source
    assert "BC_ROLLOUT_N=1" in source
    assert "SAB_CTXGRAPH_PROTOCOL=controller" in source
    assert "TRAINER_VAL_ONLY=True" in source
    assert "export TORCHDYNAMO_DISABLE=0" in source
    assert "export VLLM_USE_AOT_COMPILE=1" in source


def test_new_vllm_lora_api_uses_native_disk_loader_and_tensor_compatibility():
    source = _read("verl/utils/vllm/utils.py")
    assert "from vllm.lora.lora_model import LoRAModel" in source
    assert "return native_load_adapter(self, lora_request)" in source
    assert "model_vocab_size=self.vocab_size" in source
    assert 'if "model_vocab_size" not in str(exc)' in source
    preflight = _read("scripts/check_vllm_eval_compat.py")
    assert "VLLMHijack.hijack()" in preflight
    assert 'importlib.import_module("verl.workers.rollout.vllm_rollout.vllm_rollout")' in preflight
    assert "validate_worker_wrapper_constructor()" in preflight
    assert "validate_qwen35_aot_configuration()" in preflight
    assert "Qwen3.5/3.6 vLLM AOT startup requires TORCHDYNAMO_DISABLE=0" in preflight
    assert "vLLM eval/LoRA/rollout imports, AOT config, and worker constructor: ok" in preflight


def test_qwen36_eval_can_override_legacy_eager_runner_defaults():
    for path in (
        "scripts/eval_bc_baseline_8b_4node_zeroshot.sh",
        "scripts/train_bc_ctxgraph_8b_paperfaithful_5node_48h.sh",
        "scripts/eval_sab_react_30b_instruct_8node_smoke.sh",
    ):
        source = _read(path)
        assert "export TORCHDYNAMO_DISABLE=${TORCHDYNAMO_DISABLE:-1}" in source


def test_fp8_private_api_mismatch_is_lazy_for_bf16_rollouts():
    source = _read("verl/utils/vllm/vllm_fp8_utils.py")
    import_block = source.split("try:", 1)[1].split("logger =", 1)[0]
    assert 'raise ImportError("FP8 quantization not available")' not in import_block
    assert "_FP8_IMPORT_ERROR = e" in import_block
    assert 'if getattr(vllm_config, "quant_config", None) is None:' in source
    assert "_require_fp8_support()" in source


def test_vllm_worker_wrapper_supports_legacy_and_current_constructors():
    source = _read("verl/workers/rollout/vllm_rollout/vllm_rollout.py")
    assert "inspect.signature(WorkerWrapperBase).parameters" in source
    assert 'if "vllm_config" in parameters:' in source
    assert 'if "rpc_rank" in parameters:' in source
    assert 'kwargs["global_rank"] = 0' in source
    assert "WorkerWrapperBase(**_worker_wrapper_init_kwargs(self.vllm_config))" in source
    assert "from vllm.v1.serial_utils import run_method as vllm_run_method" in source
    assert "return vllm_run_method(self.inference_engine, method, args, kwargs)" in source


def test_browsecomp_runners_isolate_search_and_trainer_dataset_caches():
    for path in (
        "scripts/eval_bc_baseline_8b_4node_zeroshot.sh",
        "scripts/train_bc_ctxgraph_8b_paperfaithful_5node_48h.sh",
    ):
        source = _read(path)
        assert "export HF_DATASETS_CACHE=$HF_HOME/datasets" in source
        assert 'if [ "$CONDA_ENV_NAME" = "deepseek_v4" ]' in source
        assert 'export LD_PRELOAD="$LIBGOMP_PATH:$TORCH_GLOBAL_DEPS_PATH"' in source
