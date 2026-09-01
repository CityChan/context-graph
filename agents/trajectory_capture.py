"""Dependency-light helpers for capturing complete multi-agent policy chats."""

from __future__ import annotations

import copy


def serialize_agent_trajectories(agent):
    """Snapshot every main/branch chat for complete-policy distillation."""
    return [
        {
            "agent_name": name,
            "is_main": name == "main",
            "messages": copy.deepcopy(instance.messages()),
        }
        for name, instance in agent.items()
    ]
