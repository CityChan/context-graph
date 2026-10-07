"""ScienceWorld environment adapter for FoldAgent and ContextGraph."""

from __future__ import annotations

import collections
import json
import re
from pathlib import Path
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
        max_steps = getattr(plugin, "scienceworld_max_steps", 100)
        self._max_steps = 100 if max_steps is None else int(max_steps)
        if self._max_steps < 1:
            raise ValueError("scienceworld_max_steps must be positive")
        # Opt-in two-phase commit for the irreversible `focus on` (same for every method; default off).
        self._commit_check = bool(getattr(plugin, "scienceworld_commit_check", False))
        # Recheck variant: a confirmation is only valid while the visible room state is unchanged.
        self._commit_recheck = bool(getattr(plugin, "scienceworld_commit_recheck", False))
        self._look = ""
        self.commit_evidence = None  # optional callable(target) -> str supplied by a memory system
        self._checked_commits = {}
        self._task_description = ""

    @staticmethod
    def _scalar(value):
        try:
            import numpy as np
            array = np.asarray(value)
            return array.item() if array.ndim == 0 else array[0]
        except Exception:
            return value

    async def init_env(self, item):
        self.close()
        self.stats.clear()
        extra_info = self._scalar(item.non_tensor_batch["extra_info"])
        if isinstance(extra_info, str):
            extra_info = json.loads(extra_info)
        self.instance_info = dict(extra_info or {})
        self.env_fail = False
        self.is_finish = False
        self.finish = False
        self._completed = False
        self._step_count = 0
        self._checked_commits = {}
        self._look = ""

        task_name = str(self.instance_info.get("task_name", "")).strip()
        variation_idx = int(self.instance_info.get("variation_idx", 0))
        simplification = str(self.instance_info.get("simplification", ""))
        if not task_name:
            self.env_fail = True
            raise ValueError("ScienceWorld task_name is required")

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
            self._look = str((info or {}).get("look", observation))
            task_description = self._env.get_task_description()
            self._task_description = str(task_description)
            self.instance_info["problem_statement"] = (
                f"{task_description}\n\nInitial observation:\n{observation}"
            )
            self.stats["variation_idx"] = variation_idx
            self.stats["environment_score"] = float((info or {}).get("score", 0))
            self._audit({"event": "reset", "task_description": task_description,
                         "observation": observation, "info": info})
        except Exception as exc:
            print(f"[ScienceWorld] Env init failed: {exc}")
            self.stats["env_init_error"] += 1
            self.env_fail = True
            self.close()
            raise RuntimeError("ScienceWorld initialization failed") from exc

    async def run_action(self, response: str) -> dict | None:
        if self.is_finish or self.env_fail:
            return {"action": "finish", "observation": "This episode has ended."}
        self.stats["action"] += 1
        fn_call = self._parse_fn_call(response)
        if fn_call is None or fn_call["function"] != "action":
            self.stats["invalid_tool"] += 1
            return {"observation": "Use exactly one action tool call with a command parameter."}

        command = fn_call["arguments"].get("command", "").strip()
        if not command:
            self.stats["empty_command"] += 1
            return {"observation": "The action command cannot be empty."}
        if self._commit_check:
            check = self._commit_gate(command)
            if check is not None:
                return {"observation": check}

        try:
            observation, reward, completed, info = self._env.step(command)
            self._step_count += 1
            self.stats["environment_steps"] = self._step_count
            # ScienceWorld's `valid` is a list of available actions, not a validity flag.
            info = info or {}
            self._look = str(info.get("look", self._look))
            self.stats["environment_moves"] = int(info.get("moves", self._step_count))
            score = float(info["score"])
            self.stats["environment_score"] = score
            self._audit({"event": "step", "command": command, "observation": observation,
                         "reward": reward, "done": bool(completed), "info": info})
            self._completed = score >= 100
            step_limit = self._step_count >= self._max_steps
            if completed or self._completed or step_limit:
                self.is_finish = True
                self.finish = True
                self.stats["completed"] = int(self._completed)
                self.stats["environment_terminated"] = 1
                self.stats["environment_step_limit"] = int(step_limit and not self._completed)
                self.stats["environment_failure"] = int(score < 0)
                message = "Task completed successfully." if self._completed else f"Episode ended with score {score:g}/100."
                observation = f"{observation}\n\n{message}"
                return {"action": "finish", "observation": str(observation)}
            return {"observation": str(observation)}
        except Exception as exc:
            print(f"[ScienceWorld] Action failed: {exc}")
            self.stats["env_error"] += 1
            self.env_fail = True
            self.is_finish = self.finish = True
            self.close()
            raise RuntimeError("ScienceWorld simulator action failed") from exc

    def _commit_gate(self, command):
        """The first `focus on X` for a target in an episode returns a confirmation request instead of a
        simulator step; sending the same command again (at any later point) commits it. With
        ``scienceworld_commit_recheck`` the confirmation expires once the visible room state changes."""
        norm = " ".join(command.lower().split())
        if not norm.startswith("focus on "):
            return None
        if norm in self._checked_commits and (
                not self._commit_recheck or self._checked_commits[norm] == self._look):
            self.stats["commit_confirmed"] += 1
            return None
        self.stats["commit_rechecks"] += int(norm in self._checked_commits)
        self._checked_commits[norm] = self._look
        self.stats["commit_checks"] += 1
        target = command.strip()[len("focus on"):].strip()
        message = (f"[Commit check] No simulator step was taken. `focus on {target}` is irreversible: focusing on "
                   f"an object that is not the requested target fails the task immediately.\n"
                   f"Task: {self._task_description}\n")
        evidence = self.commit_evidence(target) if self.commit_evidence is not None else ""
        if evidence:
            message += evidence + "\n"
        message += ("If the observations show that this object is exactly the target the task asks for, send the "
                    "same command again to commit. Otherwise choose a different action.")
        self._audit({"event": "commit_check", "command": command, "observation": message})
        return message

    async def get_reward(self, item, messages, context) -> tuple:
        reward = 0.0 if self.env_fail else float(self._completed)
        self.stats["task_reward"] = reward
        self.stats["completed"] = int(self._completed)
        return ("", reward, {"ans_reward": reward, "task_complete": int(self._completed)})

    def _audit(self, record):
        path = self.instance_info.get("tool_log")
        if path:
            with Path(path).open("a", encoding="utf8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

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
