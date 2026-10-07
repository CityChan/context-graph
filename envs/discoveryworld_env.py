"""Text-only adapter using the pinned official UI/action/tick implementation."""
from __future__ import annotations

import collections
import json
import math
from pathlib import Path
import re
import tempfile

from envs.discoveryworld_protocol import tasks_for
from envs.discoveryworld_observation import PROFILES, dumps, encode
from envs.scienceworld_env import ScienceWorldEnv


class DiscoveryWorldEnv(ScienceWorldEnv):
    def __init__(self, config, tokenizer, ability):
        self.config, self.tokenizer, self.ability = config, tokenizer, ability
        self.stats = collections.Counter()
        self.instance_info = {}
        self.env_fail = self.is_finish = self.finish = self._completed = False
        self._env = self._frames = None
        self._step_count = 0
        self._max_steps = int(getattr(config.plugin, "discoveryworld_max_steps", 100))
        self._observation_profile = getattr(config.plugin, "discoveryworld_observation_profile", "full")
        if self._observation_profile not in PROFILES:
            raise ValueError("Unknown DiscoveryWorld observation profile")
        if self._max_steps < 1:
            raise ValueError("discoveryworld_max_steps must be positive")

    async def init_env(self, item):
        self.close()
        self.stats.clear()
        extra = self._scalar(item.non_tensor_batch["extra_info"])
        self.instance_info = dict(json.loads(extra) if isinstance(extra, str) else extra)
        self.env_fail = self.is_finish = self.finish = self._completed = False
        self._step_count = 0
        try:
            from discoveryworld.DiscoveryWorldAPI import DiscoveryWorldAPI
            task = {k: self.instance_info[k] for k in ("task_id", "scenario", "difficulty", "seed")}
            if task not in tasks_for("all"):
                raise ValueError("Unknown public DiscoveryWorld task")
            self._env = DiscoveryWorldAPI()
            self._frames = tempfile.TemporaryDirectory(prefix="discoveryworld-frames-")
            self._env.FRAME_DIR = self._frames.name + "/"
            if not self._env.loadScenario(task["scenario"], task["difficulty"], task["seed"], numUserAgents=1):
                raise RuntimeError("Official scenario load failed")
            self._actions = self._env.listKnownActions(limited=False)
            observation = self._observe()
            self._score()
            self.instance_info["problem_statement"] = dumps({
                "actions": self._actions, "teleport_locations": self._env.listTeleportLocationsDict(),
                "initial_observation": encode(observation, self._observation_profile)}, self._observation_profile)
            self._audit({"event": "reset", "observation": observation})
        except Exception:
            self.stats["env_init_error"] += 1
            self.env_fail = True
            self.close()
            raise

    def _observe(self):
        # Exact text component of getAgentObservation; skip only image rendering/PNG writes.
        ui = self._env.ui[0].renderJSON()
        ui["taskProgress"] = [{k: task[k] for k in ("taskName", "description", "completed", "completedSuccessfully")
                               if k in task} for task in ui.get("taskProgress", [])]
        self._env.taskProgress = ui["taskProgress"]
        self._env.steps = ui["world_steps"]
        return ui

    def _score(self):
        # Oracle scorecards belong ONLY to grading artifacts, never model messages.
        cards = self._env.getTaskScorecard()
        if not cards:
            raise RuntimeError("Missing official scorecard")
        scores = [float(c["scoreNormalized"]) for c in cards]
        if any(not math.isfinite(s) or not 0 <= s <= 1 for s in scores):
            raise RuntimeError("Invalid official normalized score")
        self.stats["environment_score"] = 100 * sum(scores) / len(scores)
        self._completed = all(c["completedSuccessfully"] for c in cards)
        self.stats["completed"] = int(self._completed)
        path = self.instance_info.get("grading_log")
        if path:
            Path(path).write_text(json.dumps(cards, ensure_ascii=False, indent=2), encoding="utf8")
        return all(c["completed"] for c in cards)

    def _command(self, response):
        matches = re.findall(r"<function=action>\s*<parameter=command>(.*?)</parameter>\s*</function>", response, re.S)
        if len(matches) != 1:
            raise ValueError("Use one action tool whose command is a JSON object")
        command = json.loads(matches[0])
        if not isinstance(command, dict):
            raise ValueError("command must be a JSON object")
        if self._env.isAgentInDialog(0):
            if set(command) != {"chosen_dialog_option_int"} or type(command["chosen_dialog_option_int"]) is not int:
                raise ValueError("In dialog use only chosen_dialog_option_int with an integer option")
        else:
            action = command.get("action")
            if not isinstance(action, str) or action not in self._actions:
                raise ValueError("Use an action from the supplied action catalogue")
            required = set(self._actions[action]["args"])
            if not required <= command.keys() or not command.keys() <= {"action", "arg1", "arg2"}:
                raise ValueError("Incorrect action arguments; follow the action catalogue")
            for key in required:
                expected = str if action in ("MOVE_DIRECTION", "ROTATE_DIRECTION", "TELEPORT_TO_LOCATION") else int
                if type(command[key]) is not expected:
                    raise ValueError(f"{key} must be {expected.__name__}")
        return command

    async def run_action(self, response):
        if self.is_finish or self.env_fail:
            return {"action": "finish", "observation": "Episode ended."}
        self.stats["action"] += 1
        try:
            command = self._command(response)
        except (ValueError, TypeError) as exc:
            self.stats["invalid_tool"] += 1
            self._audit({"event": "invalid_action", "response": response, "error": str(exc)})
            return {"observation": str(exc)}
        try:
            result = self._env.performAgentAction(0, command)
            tick = self._env.tick()
            if not tick.get("success"):
                raise RuntimeError(f"Official tick failed: {tick}")
            self._step_count += 1
            self.stats["environment_steps"] = self._step_count
            self.stats["invalid_environment_actions"] += int(not result["success"])
            observation = self._observe()
            ended = self._score()
            limit = self._step_count >= self._max_steps
            self.is_finish = self.finish = ended or limit
            self.stats["environment_terminated"] = int(ended)
            self.stats["environment_step_limit"] = int(limit and not ended)
            self._audit({"event": "step", "command": command, "result": result,
                         "observation": observation, "done": self.is_finish})
            payload = {"observation": dumps({"action_result": result,
                       "ui": encode(observation, self._observation_profile)}, self._observation_profile)}
            if self.is_finish:
                payload["action"] = "finish"
            return payload
        except Exception:
            self.stats["env_error"] += 1
            self.env_fail = self.is_finish = self.finish = True
            raise

    def close(self):
        if self._env is not None:
            import pygame
            pygame.quit()
            self._env = None
        if self._frames is not None:
            self._frames.cleanup()
            self._frames = None
