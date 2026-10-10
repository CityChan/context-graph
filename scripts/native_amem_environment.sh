#!/bin/bash
# Source from a native launcher after its simulator/agent environment is ready.
prepare_native_amem() {
    local base_python=$1 artifacts=$2 library
    export USE_TF=0 USE_TORCH=1 USE_FLAX=0 FORCE_TF_AVAILABLE=0
    export HF_HOME="$SCRATCH/hf_cache" HF_HUB_CACHE="$SCRATCH/hf_cache/hub"
    library=$("$base_python" -c 'import importlib.util,pathlib; print(pathlib.Path(importlib.util.find_spec("torch").origin).parent / "lib/libtorch_global_deps.so")')
    [[ -s "$library" ]] || { echo 'Missing native Torch dependencies'; return 2; }
    LD_PRELOAD="$library" "$base_python" scripts/prepare_amem_environment.py --root "$SCRATCH/context-graph-agent-benchmarks/envs" --python-file "$artifacts/amem-native-python.txt"
    AMEM_NATIVE_PYTHON=$(cat "$artifacts/amem-native-python.txt")
    export AMEM_NATIVE_PYTHON
    env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE LD_PRELOAD="$library" "$AMEM_NATIVE_PYTHON" scripts/prepare_amem_embedding.py
    "$AMEM_NATIVE_PYTHON" -m pip freeze --all > "$artifacts/amem-native-environment.txt"
}
