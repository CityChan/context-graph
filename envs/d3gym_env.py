"""Agent environment adapter for official D3-Gym task images."""

from __future__ import annotations

import collections
import copy
import asyncio
import os
import re
import tempfile
from typing import Any

from envs.d3gym_sandbox import D3GymSandbox
from envs.local_search import extract_fn_call
from envs.scienceagent_env import ScienceAgentEnv, _format_exec_result


class D3GymEnv(ScienceAgentEnv):
    """Expose D3-Gym images through the existing scientific code tools."""

    def __init__(self, config, tokenizer, ability):
        super().__init__(config, tokenizer, ability)
        self.stats = collections.Counter()
        self.expected_outputs: list[str] = []
        self.expected_output_basenames: list[str] = []
        self.execution_cwd = "/task"
        self.persistent_python_state = False
        self.runtime = "auto"
        self.image: str | None = None

    async def init_env(self, item):
        extra = item.non_tensor_batch.get("extra_info")
        if hasattr(extra, "ndim"):
            extra = extra.item() if extra.ndim == 0 else extra[0]
        if not isinstance(extra, dict):
            self.env_fail = True
            self.instance_info = {"problem_statement": "(missing D3-Gym task spec)"}
            return

        self.task_id = str(extra.get("task_id", "unknown"))
        self.instruction = extra.get("instruction") or extra.get("query") or ""
        self.expected_outputs = [
            str(value).replace("\\", "/").removeprefix("pred_results/")
            for value in (extra.get("expected_outputs") or [])
            if value
        ]
        self.expected_output_basenames = list(self.expected_outputs)
        self.expected_output_basename = self.expected_outputs[0] if len(self.expected_outputs) == 1 else None
        self.expected_output = self.expected_output_basename
        self.input_manifest = list(extra.get("input_paths") or [])
        self.instance_info = copy.deepcopy(extra)
        self.instance_info["problem_statement"] = self.instruction

        root = os.environ.get("D3GYM_WORKDIR_ROOT")
        if root:
            os.makedirs(root, exist_ok=True)
            self.workdir = tempfile.mkdtemp(prefix=f"d3gym_{self.task_id}_", dir=root)
        else:
            self.workdir = tempfile.mkdtemp(prefix=f"d3gym_{self.task_id}_")

        self.runtime = str(extra.get("runtime") or os.environ.get("D3GYM_RUNTIME") or "auto")
        self.image = extra.get("image")
        try:
            self.sandbox = await asyncio.to_thread(
                D3GymSandbox,
                task_id=self.task_id,
                workdir=self.workdir,
                image=self.image,
                runtime=self.runtime,
                task_dir=extra.get("task_dir"),
                per_call_timeout=float(getattr(getattr(self.config, "plugin", object()), "sandbox_timeout", 120.0)),
                eval_timeout=float(getattr(getattr(self.config, "plugin", object()), "eval_timeout", 300.0)),
                eval_script_source=extra.get("eval_script"),
            )
            self.runtime = self.sandbox.runtime
            discovered_inputs = await asyncio.to_thread(self.sandbox.list_input_files)
            if discovered_inputs:
                self.input_manifest = discovered_inputs
        except Exception as exc:  # noqa: BLE001 - report through trajectory, don't kill Ray worker
            self.env_fail = True
            self.sandbox = None
            self.stats["env_error"] += 1
            self.instance_info["problem_statement"] += (
                "\n\n[D3-Gym environment unavailable: "
                f"{type(exc).__name__}: {exc}]"
            )
            print(f"[D3-Gym env] task={self.task_id} init failed: {type(exc).__name__}: {exc}")
            if os.environ.get("D3GYM_STRICT_INIT", "0") == "1":
                raise RuntimeError(
                    f"D3-Gym strict initialization failed for {self.task_id}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc

    async def run_action(self, response: str) -> dict:
        self.stats["action"] += 1
        if self.env_fail or self.sandbox is None:
            return {"observation": "[Error] D3-Gym environment is unavailable."}
        calls = extract_fn_call(response)
        if not calls:
            return {"observation": "No function call was detected in the model response."}

        observation = ""
        for call in calls:
            name = call.get("function", "")
            args = call.get("arguments", {}) or {}
            if name == "python_exec":
                self.stats["python_exec"] += 1
                code = args.get("code", "")
                if not code:
                    observation += '[Error] The "python_exec" function requires a "code" argument.\n'
                    continue
                result = await asyncio.to_thread(self.sandbox.execute, code)
                observation += _format_exec_result(result)
                observation += self._format_output_status()
            elif name == "finish":
                produced = self.sandbox.list_output_files()
                if not produced:
                    self.stats["finish_rejected"] += 1
                    return {
                        "observation": (
                            "[finish rejected] pred_results/ is empty. Create the requested "
                            "artifacts before finishing.\n" + self._format_output_status().strip()
                        )
                    }
                self.stats["finish"] += 1
                self.stats["is_finish"] = 1
                self.is_finish = True
                self.predicted_answer = (args.get("message", ""), produced, self.expected_outputs)
                return {"action": "finish"}
            else:
                observation += f'[Error] The function "{name}" is not handled by D3GymEnv.\n'
        return {"observation": observation.strip()}

    async def get_reward(self, item, messages, context):
        if self.env_fail or self.sandbox is None:
            return "D3-Gym environment unavailable", 0.0, {"env_error": 1}
        produced = self.sandbox.list_output_files()
        if not produced:
            return "no output files", 0.0, {"produced_files": 0, "valid_execution": 0}
        result = await asyncio.to_thread(self.sandbox.evaluate)
        score = float(result["score"])
        print(
            f"[D3-Gym eval] task={self.task_id} score={score:.3f} "
            f"rule={result['rule']} runtime={self.runtime} produced={len(produced)} :: "
            f"{str(result['detail'])[:300]}"
        )
        return str(result["detail"]), score, {
            "produced_files": len(produced),
            "valid_execution": 1,
            "eval_rule": result["rule"],
        }

    def _format_output_status(self) -> str:
        if self.sandbox is None:
            return "\n[output_status] D3-Gym sandbox unavailable\n"
        produced = self.sandbox.list_output_files()
        if not self.expected_outputs:
            return f"\n[output_status] produced={produced}\n"
        missing = [path for path in self.expected_outputs if path not in produced]
        return (
            f"\n[output_status] expected={self.expected_outputs} "
            f"missing={missing} produced={produced}\n"
        )


def extract_input_paths(instruction: str) -> list[str]:
    """Extract task-image paths for the prompt's working-environment block."""
    instruction = instruction or ""
    paths: set[str] = set()
    # Prefer quoted/backticked paths so filenames containing spaces survive.
    for quoted in re.findall(r"[`'\"]([^`'\"]*(?:benchmark/)?datasets/[^`'\"]+)[`'\"]", instruction):
        start = quoted.find("benchmark/datasets/")
        if start < 0:
            start = quoted.find("datasets/")
        paths.add(quoted[start:].rstrip(".,;: )]"))
    # Unquoted paths stop at whitespace.  This avoids swallowing the prose
    # following a path, which the old character class allowed.
    paths.update(
        match.rstrip(".,;: )]")
        for match in re.findall(r"(?:benchmark/)?datasets/[A-Za-z0-9_./+-]+", instruction)
    )
    return sorted(
        path for path in paths
        if path and not any(other != path and other.startswith(path + " ") for other in paths)
    )
