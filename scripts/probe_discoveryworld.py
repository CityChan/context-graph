"""Run real simulator startup/action checks without loading a language model."""
import argparse
import asyncio
import contextlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from envs.discoveryworld_env import DiscoveryWorldEnv
from envs.discoveryworld_protocol import tasks_for, verify_install


async def probe(output, difficulty):
    output.mkdir(parents=True, exist_ok=True)
    provenance = verify_install()
    records = []
    # Seed zero for every requested scenario/difficulty; this is plumbing, not performance.
    for task in [t for t in tasks_for(difficulty) if t["seed"] == 0]:
        directory = output / task["task_id"]
        directory.mkdir(exist_ok=True)
        env = DiscoveryWorldEnv(SimpleNamespace(plugin=SimpleNamespace(discoveryworld_max_steps=2)), None, "DiscoveryWorld@real")
        with (directory / "simulator.log").open("w", encoding="utf8") as log, contextlib.redirect_stdout(log):
            try:
                item = SimpleNamespace(non_tensor_batch={"extra_info": dict(task,
                    tool_log=str(directory / "tools.jsonl"), grading_log=str(directory / "scorecard.json"))})
                await env.init_env(item)
                start = env._env.steps
                for _ in range(2):
                    result = await env.run_action('<function=action><parameter=command>{"action":"ROTATE_DIRECTION","arg1":"north"}</parameter></function>')
                if env._env.steps != start + 2 or result.get("action") != "finish" or env.env_fail:
                    raise RuntimeError("Simulator did not advance/terminate correctly")
                records.append({"task_id": task["task_id"], "passed": True, "environment_steps": 2})
            finally:
                env.close()
    report = {"simulator": provenance, "purpose": "real environment mechanics only; no model performance evaluation", "cases": records}
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    print("DISCOVERYWORLD_SIMULATOR_OK " + json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--difficulty", choices=["Easy", "Normal", "Challenge", "all"], default="Normal")
    args = parser.parse_args()
    asyncio.run(probe(args.output, args.difficulty))
