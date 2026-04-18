"""ALFWorld TextWorld environment for FoldAgent / ContextGraph.

Uses real ALFWorld game files via TextWorld gym.
Requires: pip install textworld alfworld

Ability flags (parsed from the `ability` string):
  - "hard"  → MemexRL-style: hide admissible_commands from the agent

Agent action format:
  <function=action><parameter=command>go to desk 1</parameter></function>
"""

import json
import os
import re
import collections
import tempfile
from contextlib import nullcontext as _nullcontext

# Global file lock for TextWorld (tatsu PDDL parser is not thread-safe)
try:
    from filelock import FileLock
    _TEXTWORLD_LOCK_PATH = os.path.join(tempfile.gettempdir(), "textworld_parser.lock")
    _TEXTWORLD_LOCK = FileLock(_TEXTWORLD_LOCK_PATH, timeout=300)
except ImportError:
    _TEXTWORLD_LOCK = _nullcontext()


class ALFWorldEnv:
    """Async wrapper around real ALFWorld (TextWorld) for the training pipeline."""

    def __init__(self, config, tokenizer, ability):
        self.config = config
        self.tokenizer = tokenizer
        self.ability = ability
        self.stats = collections.Counter()
        self.env_fail = False
        self.instance_info = {}
        self.is_finish = False
        self.finish = False

        # "hard" in ability → MemexRL-style: hide admissible commands
        self._hide_admissible = "hard" in ability.lower()

        # TextWorld state
        self._tw_env = None
        self._game_file = None
        self._admissible_commands = []
        self._won = False
        self._step_count = 0
        self._max_steps = 50
        self._look_used = False  # MemexRL: limit "look" to once per episode

    async def init_env(self, item):
        """Initialize environment from task instance."""
        import numpy as np

        def _get(arr):
            v = np.asarray(arr)
            return v.item() if v.ndim == 0 else v[0]

        extra_info = _get(item.non_tensor_batch['extra_info'])
        if isinstance(extra_info, str):
            extra_info = json.loads(extra_info)

        self.instance_info = dict(extra_info)
        self._step_count = 0
        self._won = False
        self.is_finish = False
        self.finish = False

        self._game_file = extra_info.get('game_file', '')
        if not self._game_file or not os.path.exists(self._game_file):
            print(f"[ALFWorld] Game file not found: {self._game_file}")
            self.env_fail = True
            return

        try:
            import textworld
            import textworld.gym

            lock_ctx = _TEXTWORLD_LOCK if not os.environ.get("ALFWORLD_NO_LOCK") else _nullcontext()
            with lock_ctx:
                request_infos = textworld.EnvInfos(
                    won=True,
                    admissible_commands=True,
                )
                env_id = textworld.gym.register_games(
                    [self._game_file],
                    request_infos,
                    batch_size=1,
                    max_episode_steps=self._max_steps,
                )
                self._tw_env = textworld.gym.make(env_id)
                obs_tuple, info = self._tw_env.reset()

            obs_text = obs_tuple[0] if isinstance(obs_tuple, (tuple, list)) else str(obs_tuple)
            self._admissible_commands = self._extract_admissible(info)

            task_desc = self._extract_task(obs_text)
            self.instance_info['problem_statement'] = task_desc or extra_info.get('task_desc', '')
            # MemexRL style: hide initial room description (location IDs).
            # Agent must call "look" to see surroundings.
            self.instance_info['_initial_obs'] = task_desc or obs_text

        except Exception as e:
            print(f"[ALFWorld] Env init failed: {e}")
            self.env_fail = True

    async def run_action(self, response: str) -> dict:
        """Execute agent's action."""
        self.stats['action'] += 1
        self._step_count += 1

        fn_call = self._parse_fn_call(response)
        if fn_call is None:
            return self._obs_with_commands('No valid action detected. Use the "action" tool.')

        func = fn_call['function']
        args = fn_call['arguments']

        if func == 'finish':
            self.is_finish = True
            self.finish = True
            return {'action': 'finish'}

        if func == 'think':
            return self._obs_with_commands('OK.')

        if func == 'action':
            command = args.get('command', '').strip()
            return self._step(command)

        return self._obs_with_commands(f'Unknown tool "{func}". Use "action" tool.')

    def _step(self, command: str) -> dict:
        """Execute action in TextWorld environment."""
        # MemexRL: limit "look" to once per episode
        if command.strip().lower() == "look":
            if self._look_used:
                return self._obs_with_commands(
                    "You have already used 'look' in this episode. "
                    "Try to remember the locations you saw, or explore by going to specific locations."
                )
            self._look_used = True

        try:
            lock_ctx = _TEXTWORLD_LOCK if not os.environ.get("ALFWORLD_NO_LOCK") else _nullcontext()
            with lock_ctx:
                obs_tuple, rewards, dones, info = self._tw_env.step([command])

            obs_text = obs_tuple[0] if isinstance(obs_tuple, (tuple, list)) else str(obs_tuple)
            done = dones[0] if isinstance(dones, (tuple, list)) else dones
            self._admissible_commands = self._extract_admissible(info)

            won_list = info.get("won", [False]) if isinstance(info, dict) else [False]
            self._won = won_list[0] if isinstance(won_list, (tuple, list)) else won_list

            if done or self._won:
                self.is_finish = True
                self.finish = True
                if self._won:
                    return {'observation': obs_text + '\n\nTask completed successfully!'}
                return {'observation': obs_text}

            if self._step_count >= self._max_steps:
                return self._obs_with_commands(obs_text + '\n\nMax steps reached.')

            self.stats[command.split()[0] if command else 'unknown'] += 1
            print(f'[ALF] step={self._step_count} cmd="{command[:60]}" won={self._won}')
            return self._obs_with_commands(obs_text)

        except Exception as e:
            print(f'[ALF] Error: {e}')
            return self._obs_with_commands(f'Error executing action: {e}')

    def _obs_with_commands(self, obs_text: str) -> dict:
        """Return observation with admissible commands (unless hidden in hard mode)."""
        if self._hide_admissible:
            return {'observation': obs_text}
        if self._admissible_commands:
            cmd_str = ", ".join(self._admissible_commands)
            obs_text += f"\n\nAdmissible commands: [{cmd_str}]"
        return {'observation': obs_text}

    def _extract_admissible(self, info) -> list:
        """Extract admissible commands from TextWorld info dict."""
        if isinstance(info, dict):
            admissible = info.get("admissible_commands", [[]])
            return admissible[0] if admissible else []
        return []

    def _extract_task(self, obs: str) -> str:
        """Extract task description from initial observation."""
        for line in obs.split("\n"):
            if "task is to" in line.lower() or "your task" in line.lower():
                return line.strip()
        return ""

    def _parse_fn_call(self, text):
        if text is None:
            return None
        func_matches = re.findall(r'<function=([^>]+)>', text)
        if not func_matches:
            return None
        last_function = func_matches[-1]
        last_func_pos = text.rfind(f'<function={last_function}>')
        text_after = text[last_func_pos:]
        params = dict(re.findall(r'<parameter=([^>]+)>(.*?)</parameter>', text_after, re.DOTALL))
        return {'function': last_function, 'arguments': params}

    async def get_reward(self, item, messages, context) -> tuple:
        if self.env_fail:
            return ("", 0, {"ans_reward": 0.0})
        reward = 1.0 if self._won else 0.0
        return ("", reward, {"ans_reward": reward, "task_complete": int(reward > 0)})

    def close(self):
        if self._tw_env is not None:
            try:
                self._tw_env.close()
            except Exception:
                pass
            self._tw_env = None

    def __del__(self):
        self.close()
