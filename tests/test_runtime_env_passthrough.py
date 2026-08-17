from pathlib import Path


def test_ppo_runtime_env_preserves_linker_preload():
    source = Path("verl/trainer/constants_ppo.py").read_text(encoding="utf-8")
    passthrough_start = source.index("# Preserve CUDA and compiler paths")
    passthrough_end = source.index("return runtime_env", passthrough_start)
    passthrough = source[passthrough_start:passthrough_end]

    assert '"LD_PRELOAD",' in passthrough


def test_qwen35_smoke_preloads_torch_dependencies_before_ray():
    source = Path("scripts/eval_sab_react_30b_instruct_8node_smoke.sh").read_text(encoding="utf-8")
    preload = "export LD_PRELOAD=${LD_PRELOAD}:$TORCH_GLOBAL_DEPS_PATH"

    assert "libtorch_global_deps.so" in source
    assert preload in source
    assert source.index(preload) < source.index("ray start --head")
