"""Dependency-light helpers for capturing complete multi-agent policy chats."""

from __future__ import annotations

import copy


def serialize_agent_trajectories(agent):
    """Snapshot every main/branch chat for complete-policy distillation."""
    trajectories = []
    for name, instance in agent.items():
        sessions = (instance.session_views() if getattr(instance, "summary_sessions", None)
                    else [(None, 0, instance)])
        for index, start, view in sessions:
            row = {
                "agent_name": name,
                "is_main": name == "main",
                "messages": copy.deepcopy(view.messages()),
            }
            if index is not None:
                row.update(summary_session_index=index, archive_start_turn=start)
            trajectories.append(row)
    return trajectories
