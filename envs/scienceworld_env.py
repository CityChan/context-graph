"""ScienceWorld environment adapter for FoldAgent and ContextGraph."""

from __future__ import annotations

import collections
import json
import re
from typing import Any


class ScienceWorldEnv:
    """Async-compatible wrapper around the local ScienceWorld JVM simulator."""

    def __init__(self, config, tokenizer, ability):
        self.config = config
        self.tokenizer = tokenizer
        self.ability = ability
        self.stats = collections.Counter()
        self.instance_info: dict[str, Any] = {}
        self.env_fail = False
        self.is_finish = False
        self.finish = False
        self._env = None
        self._completed = False
        self._step_count = 0
        plugin = getattr(config, "plugin", None)
        self._max_steps = int(getattr(plugin, "scienceworld_max_steps", 100) or 100)

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
        self._completed = False
        self._step_count = 0

        task_name = str(self.instance_info.get("task_name", "")).strip()
        variation_idx = int(self.instance_info.get("variation_idx", 0))
        simplification = str(self.instance_info.get("simplification", ""))
        if not task_name:
            self.env_fail = True
            return

        try:
            from scienceworld import ScienceWorldEnv as Simulator

            self._env = Simulator(taskName=task_name, envStepLimit=self._max_steps)
            self._env.load(
                taskName=task_name,
                variationIdx=variation_idx,
                simplificationStr=simplification,
                generateGoldPath=False,
            )
            observation, info = self._env.reset()
            task_description = self._env.get_task_description()
            self.instance_info["problem_statement"] = (
                f"{task_description}\n\nInitial observation:\n{observation}"
            )
            self.stats["variation_idx"] = variation_idx
            self.stats["initial_valid"] = int(bool((info or {}).get("valid", True)))
        except Exception as exc:
            print(f"[ScienceWorld] Env init failed: {exc}")
            self.stats["env_init_error"] += 1
            self.env_fail = True

    async def run_action(self, response: str) -> dict | None:
        self.stats["action"] += 1
        fn_call = self._parse_fn_call(response)
        if fn_call is None or fn_call["function"] != "action":
            self.stats["invalid_tool"] += 1
            return {"observation": "Use exactly one action tool call with a command parameter."}

        command = fn_call["arguments"].get("command", "").strip()
        if not command:
            self.stats["empty_command"] += 1
            return {"observation": "The action command cannot be empty."}

        try:
            observation, reward, completed, info = self._env.step(command)
            self._step_count += 1
            self.stats["environment_steps"] = self._step_count
            self.stats["invalid_actions"] += int(not bool((info or {}).get("valid", True)))
            self.stats["environment_score"] = float((info or {}).get("score", reward) or 0.0)
            if completed:
                self._completed = True
                self.is_finish = True
                self.finish = True
                self.stats["completed"] = 1
                observation = f"{observation}\n\nTask completed successfully."
            return {"observation": str(observation)}
        except Exception as exc:
            print(f"[ScienceWorld] Action failed: {exc}")
            self.stats["env_error"] += 1
            return {"observation": f"Environment error: {exc}"}

    async def get_reward(self, item, messages, context) -> tuple:
        reward = 0.0 if self.env_fail else float(self._completed)
        self.stats["task_reward"] = reward
        self.stats["completed"] = int(self._completed)
        return ("", reward, {"ans_reward": reward, "task_complete": int(self._completed)})

    @staticmethod
    def _parse_fn_call(text: str | None) -> dict | None:
        if not text:
            return None
        matches = list(re.finditer(r"<function=([^>]+)>", text))
        if not matches:
            return None
        match = matches[-1]
        function = match.group(1).strip()
        tail = text[match.end():]
        arguments = {}
        for param in re.finditer(
            r"<parameter=([^>]+)>(.*?)(?:</parameter>|(?=<parameter=)|</function>|$)",
            tail,
            re.DOTALL,
        ):
            key = param.group(1).strip()
            value = param.group(2).strip().strip('`"\'')
            if key and value:
                arguments[key] = value.splitlines()[0].strip()
        return {"function": function, "arguments": arguments}

    def close(self):
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
            self._env = None

    def __del__(self):
        self.close()
