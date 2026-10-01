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

import logging
import os
from copy import deepcopy

import torch
from omegaconf import OmegaConf


_SENSITIVE_ENV_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")


def redact_ray_init_kwargs(ray_init_kwargs):
    """Return a log-safe copy without exposing credentials in env vars."""
    if OmegaConf.is_config(ray_init_kwargs):
        redacted = OmegaConf.to_container(ray_init_kwargs, resolve=False)
    else:
        redacted = deepcopy(ray_init_kwargs)

    env_vars = redacted.get("runtime_env", {}).get("env_vars", {})
    if isinstance(env_vars, dict):
        for name in env_vars:
            upper_name = str(name).upper()
            if any(marker in upper_name for marker in _SENSITIVE_ENV_MARKERS):
                env_vars[name] = "[REDACTED]"
    return redacted


def set_basic_config(level):
    """
    This function sets the global logging format and level. It will be called when import verl
    """
    logging.basicConfig(format="%(levelname)s:%(asctime)s:%(message)s", level=level)


def log_to_file(string):
    print(string)
    if os.path.isdir("logs"):
        with open(f"logs/log_{torch.distributed.get_rank()}", "a+") as f:
            f.write(string + "\n")
