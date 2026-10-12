"""Opt-in collection guidance; no action dispatch or reference-answer access."""

PROFILE = "graph_demonstration_v1"
ASSISTED_PROFILE = "framework_assisted_v1"

PROMPT = """SFT demonstration collection (not the ordinary evaluation prompt).
Complete the task while demonstrating useful, model-chosen graph memory.
Keep reusable notes of verified findings, source/object IDs, current state,
failed attempts and remaining work. After a meaningful phase, consider whether
observations should be merged into a concise note or obsolete states pruned.
Choose the nodes and write the summary yourself from evidence you have read.
Do not invent results, discard unresolved evidence, or call tools just to meet a quota.

Plan memory use before the window is exhausted: merge/prune changes the graph
but DOES NOT remove old conversation. If the resource snapshot shows little
working-window space and a smaller compacted prompt fits, use graph_compact to
continue with the retained evidence. It is also available without a prior merge.
Then continue the real task and use those notes. Do not wait until there is no
room to request compaction. There is no automatic compaction or operation schedule.

When an action fails or changes nothing, read the feedback and inspect missing
information before repeating it. For an experiment, track the current apparatus
and connections, what was actually observed, and what remains untested. For
research, track each question constraint and supporting source IDs; refine a
query that repeatedly returns irrelevant evidence. Success requires completing
the task, not merely accumulating actions or graph nodes. Use the existing JSON
action interface; do not add prose or a separate reasoning response.
"""


def teacher_guidance(profile):
    if profile == "evaluation":
        return ""
    if profile == ASSISTED_PROFILE:
        return PROMPT.replace(
            "There is no automatic compaction or operation schedule.",
            "The framework schedules explicitly labelled maintenance checkpoints for this "
            "assisted demonstration. During a checkpoint only memory tools are available; "
            "you choose targets and content and must call graph_compact to resume the task. "
            "These demonstrations do not establish autonomous maintenance timing.")
    if profile != PROFILE:
        raise ValueError(f"Unknown SFT teacher profile: {profile}")
    return PROMPT


class AssistedMaintenance:
    """Collection-only timing assistance; never edits the graph or sampled output."""

    tools = frozenset({"merge", "prune", "graph_read", "recall", "graph_compact"})
    interval = 8
    window_fraction = 0.5
    max_requests = 6

    def __init__(self):
        self.task_actions = 0
        self.checkpoints = 0
        self.active = None

    def prepare(self, agent):
        if agent.force_action_finish:
            return
        window = agent.config.prompt_length + agent.config.response_length
        window = getattr(agent, "working_context_limit", None) or window
        tokens = len(agent.context())
        if self.active is None and self.task_actions >= 2 and (
                self.task_actions >= self.interval or tokens >= window * self.window_fraction):
            self.checkpoints += 1
            self.active = {"profile": ASSISTED_PROFILE, "checkpoint": self.checkpoints,
                           "context_uid_at_trigger": agent.context_uid,
                           "timing_source": "framework", "semantic_source": "model",
                           "trigger": "action_interval" if self.task_actions >= self.interval else "window_pressure",
                           "context_tokens_at_trigger": tokens, "requests": 0}
            agent.append({"role": "user", "content": (
                f"[Framework-assisted SFT maintenance checkpoint {self.checkpoints}; not simulator evidence]\n"
                "Pause task actions to preserve working memory. Only merge, prune, graph_read, "
                "recall and graph_compact are available during this checkpoint. You choose "
                "the operation, targets and summary from observed evidence. Retain the task, "
                "source/object IDs, current state, failed attempts and unresolved work. "
                "Do not invent facts or discard necessary evidence. Use graph_read if needed; "
                "merge related observations or prune obsolete ones only when justified. "
                "Then call graph_compact to replace history with the retained graph and "
                "resume task actions. graph_compact may be used without editing nodes. "
                f"At most {self.max_requests} model requests are available for this checkpoint; "
                "all requests and feedback count against the usual budgets.")})
        if self.active and self.active["requests"] >= self.max_requests:
            raise RuntimeError("Assisted SFT maintenance did not complete within its request limit")

    def constrain(self, schema):
        return {"anyOf": [choice for choice in schema["anyOf"]
                          if choice["properties"]["tool"]["const"] in self.tools]}

    def sampled(self, payload):
        if payload["tool"] in {"action", "search", "open_page", "python_exec"}:
            self.task_actions += 1

    def executed(self, name, success, tokens_before, tokens_after):
        if self.active and name == "graph_compact" and success and tokens_after < tokens_before:
            self.active = None
            self.task_actions = 0
