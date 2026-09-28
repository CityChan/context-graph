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

from .agent_loop import AgentLoopBase, AgentLoopManager, AgentLoopWorker, AsyncLLMServerManager
from .single_turn_agent_loop import SingleTurnAgentLoop
from .tool_agent_loop import ToolAgentLoop

# Register agent loops so Ray workers can find them. Import modules for their
# registration side effects, not classes: a training script may itself be the
# entry point and still be defining its class while this package initializes.
from scripts import train_fold, train_graph, train_baseline
# ScienceAgentBench code-domain agent loops (react_agent_code / fold_agent_code /
# context_graph_code_isolated_agent) — must be imported here so Ray agent-loop
# workers register them too, not just the driver process running train_sab.
from .code_agent_loop import (
    ReactAgentCodeLoop,
    FoldAgentCodeLoop,
    ContextGraphCodeIsolatedLoop,
)

_ = [
    SingleTurnAgentLoop,
    ToolAgentLoop,
    train_fold,
    train_graph,
    train_baseline,
    ReactAgentCodeLoop,
    FoldAgentCodeLoop,
    ContextGraphCodeIsolatedLoop,
]

__all__ = ["AgentLoopBase", "AgentLoopManager", "AsyncLLMServerManager", "AgentLoopWorker"]


def __getattr__(name):
    # Preserve explicit legacy imports without dereferencing partially
    # initialized training modules during package startup.
    exports = {"FoldAgentLoop": train_fold, "ContextGraphAgentLoop": train_graph,
               "ReactAgentLoop": train_baseline}
    if name in exports:
        return getattr(exports[name], name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
