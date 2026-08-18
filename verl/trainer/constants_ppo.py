# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os

from ray._private.runtime_env.constants import RAY_JOB_CONFIG_JSON_ENV_VAR

PPO_RAY_RUNTIME_ENV = {
    "env_vars": {
        "TOKENIZERS_PARALLELISM": "true",
        "NCCL_DEBUG": "WARN",
        "VLLM_LOGGING_LEVEL": "WARN",
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
        # symmetric memory allreduce not work properly in spmd mode
        "VLLM_ALLREDUCE_USE_SYMM_MEM": "0",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        # To prevent hanging or crash during synchronization of weights between actor and rollout
        # in disaggregated mode. See:
        # https://docs.vllm.ai/en/latest/usage/troubleshooting.html?h=nccl_cumem_enable#known-issues
        # https://github.com/vllm-project/vllm/blob/c6b0a7d3ba03ca414be1174e9bd86a97191b7090/vllm/worker/worker_base.py#L445
        "NCCL_CUMEM_ENABLE": "0",
    },
}


def get_ppo_ray_runtime_env():
    """
    A filter function to return the PPO Ray runtime environment.
    To avoid repeat of some environment variables that are already set.
    """
    working_dir = (
        json.loads(os.environ.get(RAY_JOB_CONFIG_JSON_ENV_VAR, "{}")).get("runtime_env", {}).get("working_dir", None)
    )

    runtime_env = {
        "env_vars": PPO_RAY_RUNTIME_ENV["env_vars"].copy(),
        **({"working_dir": None} if working_dir is None else {}),
    }
    for key in list(runtime_env["env_vars"].keys()):
        if os.environ.get(key) is not None:
            runtime_env["env_vars"].pop(key, None)

    # Preserve CUDA and compiler paths for Ray workers so vLLM child processes
    # can resolve CUDA shared libraries like libnvrtc at process start.
    for key in (
        "PATH",
        # Linker preloads must be present before Ray execs each worker process.
        # This is required on aarch64 when libgomp is preloaded to avoid the
        # glibc dl-tls race during the first PyTorch import in a threaded worker.
        "LD_PRELOAD",
        "LD_LIBRARY_PATH",
        "LIBRARY_PATH",
        "CPATH",
        "CUDAHOSTCXX",
        "CC",
        "CXX",
        "CUDA_HOME",
        "CUDA_ROOT",
        "CONDA_PREFIX",
        "HF_HOME",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "FLASHINFER_WORKSPACE_BASE",
        # Keep intra-allocation HTTP traffic off external proxies. Search
        # workers use LOCAL_SEARCH_URL while answer judging may still need the
        # configured proxy for public endpoints.
        "LOCAL_SEARCH_URL",
        "NO_PROXY",
        "no_proxy",
        # D3-Gym task images are launched by Ray agent-loop workers.  Preserve
        # the selected runtime and shared image/workdir caches on every node.
        "D3GYM_RUNTIME",
        "D3GYM_IMAGE_DIR",
        "D3GYM_WORKDIR_ROOT",
        "APPTAINER_CACHEDIR",
        "APPTAINER_TMPDIR",
        "SINGULARITY_CACHEDIR",
        "SINGULARITY_TMPDIR",
    ):
        value = os.environ.get(key)
        if value is not None:
            runtime_env["env_vars"][key] = value

    return runtime_env
