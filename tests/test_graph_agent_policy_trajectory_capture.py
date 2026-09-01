from agents.trajectory_capture import serialize_agent_trajectories


class FakeAgent:
    def __init__(self, messages):
        self._messages = messages

    def messages(self):
        return self._messages


def test_serializes_main_and_branch_chats_without_aliasing():
    agents = {
        "main": FakeAgent([{"role": "assistant", "content": "<function=branch>"}]),
        "#0-facts": FakeAgent([{"role": "assistant", "content": "<function=return>"}]),
    }
    snapshots = serialize_agent_trajectories(agents)
    assert [item["agent_name"] for item in snapshots] == ["main", "#0-facts"]
    assert [item["is_main"] for item in snapshots] == [True, False]
    agents["main"]._messages[0]["content"] = "changed"
    assert snapshots[0]["messages"][0]["content"] == "<function=branch>"
