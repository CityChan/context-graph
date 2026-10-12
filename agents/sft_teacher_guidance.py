"""Opt-in collection guidance; no action dispatch or reference-answer access."""

PROFILE = "graph_demonstration_v1"

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
    if profile != PROFILE:
        raise ValueError(f"Unknown SFT teacher profile: {profile}")
    return PROMPT
