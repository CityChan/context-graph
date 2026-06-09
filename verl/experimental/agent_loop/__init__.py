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

# Register agent loops so Ray workers can find them
from scripts.train_fold import FoldAgentLoop
from scripts.train_graph import ContextGraphAgentLoop
from scripts.train_baseline import ReactAgentLoop
# ScienceAgentBench code-domain agent loops (react_agent_code / fold_agent_code /
# context_graph_code_isolated_agent) — must be imported here so Ray agent-loop
# workers register them too, not just the driver process running train_sab.
from scripts.train_sab import (
    ReactAgentCodeLoop,
    FoldAgentCodeLoop,
    ContextGraphCodeIsolatedLoop,
)

_ = [
    SingleTurnAgentLoop,
    ToolAgentLoop,
    FoldAgentLoop,
    ContextGraphAgentLoop,
    ReactAgentLoop,
    ReactAgentCodeLoop,
    FoldAgentCodeLoop,
    ContextGraphCodeIsolatedLoop,
]

__all__ = ["AgentLoopBase", "AgentLoopManager", "AsyncLLMServerManager", "AgentLoopWorker"]