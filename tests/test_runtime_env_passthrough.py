from pathlib import Path


def test_ppo_runtime_env_preserves_linker_preload():
    source = Path("verl/trainer/constants_ppo.py").read_text(encoding="utf-8")
    passthrough_start = source.index("# Preserve CUDA and compiler paths")
    passthrough_end = source.index("return runtime_env", passthrough_start)
    passthrough = source[passthrough_start:passthrough_end]

    assert '"LD_PRELOAD",' in passthrough
