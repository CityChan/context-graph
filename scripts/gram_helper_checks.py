"""Semantic smoke controls for the real frozen-helper edit pipeline."""
from agents.gram_memory import GraphMemory, entity_key


async def check_helper_controls(helper):
    controls = []
    for label, text, facts in (
        ("positive", "Alice was born in Paris.", "Alice was born in Paris. Alice won a prize in 1999."),
        ("negative", "...", "Alice was born in Paris."),
    ):
        graph = GraphMemory()
        control = {"name": label}
        helper.audit({"kind": "control_start", "name": label})
        try:
            control["edit"] = await helper.edit("memory_insert", facts,
                {"id": label, "title": "Synthetic", "text": text},
                "What prize did Alice win?", graph, .9)
            control["graph"] = graph.snapshot()
            # Check the relation as well as endpoints; Alice --won_prize--> Paris
            # must not count as a passing birthplace extraction.
            birth_relations = {"born_in", "born in", "birthplace", "birth_place", "birth place",
                               "place_of_birth", "place of birth", "was_born_in", "was born in"}
            control["passed"] = (bool(graph.edges) and all(
                s == "Alice" and o == "Paris" and entity_key(r) in birth_relations
                for s, r, o in graph.edges)) if label == "positive" else not graph.edges
        except Exception as exc:
            control.update(passed=False, error=f"{type(exc).__name__}: {exc}")
        controls.append(control)
    return controls
