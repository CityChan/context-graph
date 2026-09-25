from types import SimpleNamespace

import numpy as np
import pytest

from agents.context_graph_modes import foldagent_inputs, memory_mode
from agents.prompts import create_chat


@pytest.mark.parametrize("scalar", [True, False])
def test_foldagent_control_preserves_inputs_and_restores_original_prompt(scalar):
    info = {"workflow": "search_graph", "task_id": "q1"}
    item = SimpleNamespace(non_tensor_batch={"extra_info": np.array(info if scalar else [info], dtype=object)})
    plugin = SimpleNamespace(workflow="search_graph", process_reward=["flat", "scope", "graph"], controller_owned_tool_formatting=True)
    config = SimpleNamespace(actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(plugin=plugin, response_length=1234)))
    client = object()
    context = SimpleNamespace(config=config, is_train=False, llm_client=client)
    converted, local = foldagent_inputs(item, context)
    actual = converted.non_tensor_batch["extra_info"].reshape(-1)[0]
    assert actual["task_id"] == "q1"
    assert create_chat("question", actual["workflow"]) == create_chat("question", "search_branch")
    assert info["workflow"] == "search_graph"
    assert plugin.controller_owned_tool_formatting
    assert not local.config.actor_rollout_ref.rollout.plugin.controller_owned_tool_formatting
    assert local.config.actor_rollout_ref.rollout.response_length == 1234
    assert local.llm_client is client


def test_unknown_mode_fails_closed():
    with pytest.raises(ValueError, match="Unknown"):
        memory_mode(SimpleNamespace(contextgraph_memory_mode="repaird"))
