"""Pinned public DiscoveryWorld suite; no hidden task answers or scorecards."""
import importlib.metadata
import json
import sys

REVISION = "fd591323920be0d3786ef350955de1945aa571e5"
SCENARIOS = (
    "Combinatorial Chemistry", "Archaeology Dating", "Plant Nutrients", "Reactor Lab",
    "Lost in Translation", "Space Sick", "Proteomics", "It's (not) Rocket Science!",
)
DIFFICULTIES = ("Easy", "Normal", "Challenge")
PROTOCOL = f"discoveryworld/{REVISION}/public/text-ui/shared-sequential-environment/v1"


def tasks_for(difficulty="Normal"):
    if difficulty not in (*DIFFICULTIES, "all"):
        raise ValueError("Unknown DiscoveryWorld difficulty")
    return [dict(task_id=f"discoveryworld_{SCENARIOS.index(name)}_{level.lower()}_{seed}",
                 scenario=name, difficulty=level, seed=seed)
            for seed in range(5) for name in SCENARIOS
            for level in (DIFFICULTIES if difficulty == "all" else (difficulty,))]


def verify_install():
    """Require pip's immutable VCS provenance, not just a package version."""
    if not (3, 10) <= sys.version_info[:2] < (3, 12):
        raise RuntimeError("Use a Python 3.10/3.11 private agent environment; pinned upstream uses pre-3.12 random argument conversion")
    dist = importlib.metadata.distribution("discoveryworld")
    provenance = json.loads(dist.read_text("direct_url.json") or "{}")
    if provenance.get("vcs_info", {}).get("commit_id") != REVISION:
        raise RuntimeError("Install requirements_discoveryworld.txt in a private environment; upstream commit mismatch")
    return {"version": dist.version, "revision": REVISION}
