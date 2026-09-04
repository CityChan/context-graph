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

"""JSON serialization helpers for rollout metadata."""

import os
import uuid

import numpy as np


def json_default(value):
    """Convert common rollout metadata objects to JSON-native values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (set, frozenset)):
        return list(value)
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, os.PathLike):
        return os.fspath(value)

    # Avoid importing torch in this lightweight utility. Tensor objects expose
    # this conversion chain, including CUDA tensors that must move to CPU first.
    if type(value).__module__.split(".", 1)[0] == "torch":
        detach = getattr(value, "detach", None)
        if callable(detach):
            value = detach()
        cpu = getattr(value, "cpu", None)
        if callable(cpu):
            value = cpu()
        tolist = getattr(value, "tolist", None)
        if callable(tolist):
            return tolist()

    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
