"""AppWorld environment adapter for ContextGraph trajectory generation."""

from __future__ import annotations

import collections
import json
import os
import re
from typing import Any


class AppWorldEnv:
    """Async-compatible, train-only wrapper around an AppWorld task."""

    def __init__(self, config, tokenizer, ability):
        self.config = config
        self.tokenizer = tokenizer
        self.ability = ability
        self.stats = collections.Counter()
        self.instance_info: dict[str, Any] = {}
        self.env_fail = False
        self.is_finish = False
        self.finish = False
        self._world = None
        self._task_id = ""
        self._step_count = 0

    @staticmethod
    def _scalar(value):
        try:
            import numpy as np

            array = np.asarray(value)
            return array.item() if array.ndim == 0 else array[0]
        except Exception:
            return value

    async def init_env(self, item):
        extra_info = self._scalar(item.non_tensor_batch["extra_info"])
        if isinstance(extra_info, str):
            extra_info = json.loads(extra_info)
        self.instance_info = dict(extra_info or {})
        self.env_fail = False
        self.is_finish = False
        self.finish = False
        self._step_count = 0

        self._task_id = str(self.instance_info.get("task_id", "")).strip()
        split = str(self.instance_info.get("split", "train")).strip().lower()
        if not self._task_id or split != "train":
            self.stats["rejected_non_train_task"] += int(split != "train")
            self.env_fail = True
            return

        try:
            from appworld import AppWorld

            base_name = os.environ.get("APPWORLD_EXPERIMENT_NAME", "contextgraph_sft")
            safe_task = re.sub(r"[^A-Za-z0-9_.-]+", "_", self._task_id)
            experiment_name = f"{base_name}_{os.getpid()}_{safe_task}"[:180]
            self._world = AppWorld(task_id=self._task_id, experiment_name=experiment_name)
            instruction = str(self._world.task.instruction).strip()
            app_descriptions = getattr(self._world.task, "app_descriptions", {}) or {}
            if isinstance(app_descriptions, dict) and app_descriptions:
                available = "\n".join(
                    f"- {name}: {description}" for name, description in app_descriptions.items()
                )
                instruction = f"{instruction}\n\nAvailable apps:\n{available}"
            self.instance_info["problem_statement"] = instruction
            self.stats["appworld_initialized"] = 1
        except Exception as exc:
            print(f"[AppWorld] Env init failed: {exc}")
            self.stats["env_init_error"] += 1
            self.env_fail = True

    async def run_action(self, response: str) -> dict | None:
        self.stats["action"] += 1
        fn_call = self._parse_fn_call(response)
        if fn_call is None or fn_call["function"] != "action":
            self.stats["invalid_tool"] += 1
            return {"observation": "Use exactly one action tool call with a code parameter."}
        code = fn_call["arguments"].get("code", "").strip()
        if not code:
            self.stats["empty_code"] += 1
            return {"observation": "The Python code cannot be empty."}
        try:
            observation = self._world.execute(code)
            self._step_count += 1
            self.stats["environment_steps"] = self._step_count
            if bool(self._world.task_completed()):
                self.is_finish = True
                self.finish = True
                self.stats["task_completed_signal"] = 1
                observation = f"{observation}\n\nTask completion was submitted."
            return {"observation": str(observation)}
        except Exception as exc:
            print(f"[AppWorld] Action failed: {exc}")
            self.stats["env_error"] += 1
            return {"observation": f"AppWorld execution error: {exc}"}

    def _evaluation_success(self, result: Any) -> bool:
        direct_success = getattr(result, "success", None)
        if isinstance(direct_success, bool):
            return direct_success
        if hasattr(result, "to_dict"):
            result = result.to_dict()
        if isinstance(result, bool):
            return result
        if not isinstance(result, dict):
            return False
        if isinstance(result.get("success"), bool):
            return result["success"]
        individual = result.get("individual")
        if isinstance(individual, dict):
            candidate = individual.get(self._task_id, individual)
            if isinstance(candidate, dict) and isinstance(candidate.get("success"), bool):
                return candidate["success"]
        passes = result.get("passes")
        fails = result.get("fails")
        if isinstance(passes, (list, tuple)) and isinstance(fails, (list, tuple)):
            return bool(passes) and not fails
        return False

    async def get_reward(self, item, messages, context) -> tuple:
        success = False
        if not self.env_fail and self._world is not None:
            try:
                success = self._evaluation_success(self._world.evaluate())
            except Exception as exc:
                print(f"[AppWorld] Evaluation failed: {exc}")
                self.stats["evaluation_error"] += 1
        reward = float(success)
        self.stats["task_reward"] = reward
        self.stats["appworld_success"] = int(success)
        return ("", reward, {"ans_reward": reward, "task_complete": int(success)})

    @staticmethod
    def _parse_fn_call(text: str | None) -> dict | None:
        if not text:
            return None
        matches = list(re.finditer(r"<function=([^>]+)>", text))
        if not matches:
            return None
        match = matches[-1]
        tail = text[match.end():]
        arguments = {}
        for param in re.finditer(
            r"<parameter=([^>]+)>(.*?)(?:</parameter>|(?=<parameter=)|</function>|$)",
            tail,
            re.DOTALL,
        ):
            key = param.group(1).strip()
            value = param.group(2).strip()
            if key and value:
                arguments[key] = value
        return {"function": match.group(1).strip(), "arguments": arguments}

    def close(self):
        if self._world is not None:
            try:
                self._world.close()
            except Exception:
                pass
            self._world = None

    def __del__(self):
        self.close()
