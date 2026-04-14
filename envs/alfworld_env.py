"""ALFWorld text-game environment for FoldAgent / ContextGraph.

Supports two modes:
  1. Real TextWorld mode (ability="ALFWorld@real"): Uses actual ALFWorld game files
     via TextWorld gym. Requires: pip install textworld alfworld
  2. Mock mode (ability="ALFWorld@mock"): Built-in state machine for testing
     without alfworld package.

The env provides admissible_commands each turn. Agent uses:
  <function=action><parameter=command>go to desk 1</parameter></function>
"""

import json
import os
import re
import collections
import tempfile
import random
from contextlib import nullcontext as _nullcontext
from typing import Optional

# Global file lock for TextWorld (tatsu PDDL parser is not thread-safe)
try:
    from filelock import FileLock
    _TEXTWORLD_LOCK_PATH = os.path.join(tempfile.gettempdir(), "textworld_parser.lock")
    _TEXTWORLD_LOCK = FileLock(_TEXTWORLD_LOCK_PATH, timeout=300)
except ImportError:
    _TEXTWORLD_LOCK = _nullcontext()


class ALFWorldEnv:
    """Async wrapper around ALFWorld for FoldAgent training pipeline.

    Supports real TextWorld games and mock state machine fallback.
    """

    def __init__(self, config, tokenizer, ability):
        self.config = config
        self.tokenizer = tokenizer
        self.ability = ability
        self.stats = collections.Counter()
        self.env_fail = False
        self.instance_info = {}
        self.is_finish = False
        self.finish = False

        # Determine mode from ability string
        self._use_real = "real" in ability.lower()

        # Real TextWorld state
        self._tw_env = None
        self._game_file = None
        self._admissible_commands = []
        self._won = False
        self._step_count = 0
        self._max_steps = 50
        self._look_used = False  # MemexRL: limit "look" to once per episode

        # Mock state (fallback)
        self._mock_state = None

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

        if self._use_real:
            self._init_real(extra_info)
        else:
            self._init_mock(extra_info)

    def _init_real(self, extra_info):
        """Initialize real TextWorld ALFWorld environment."""
        self._game_file = extra_info.get('game_file', '')
        if not self._game_file or not os.path.exists(self._game_file):
            print(f"[ALFWorld] Game file not found: {self._game_file}, falling back to mock")
            self._use_real = False
            self._init_mock(extra_info)
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

            # Extract task description from observation
            task_desc = self._extract_task(obs_text)
            self.instance_info['problem_statement'] = task_desc or extra_info.get('task_desc', '')
            # MemexRL style: hide initial room description (location IDs).
            # Agent must call "look" to see surroundings.
            self.instance_info['_initial_obs'] = task_desc or obs_text

        except Exception as e:
            print(f"[ALFWorld] Real env init failed: {e}, falling back to mock")
            self._use_real = False
            self._init_mock(extra_info)

    def _init_mock(self, extra_info):
        """Initialize mock state machine (no TextWorld needed)."""
        self.instance_info['problem_statement'] = extra_info.get('task_desc', '')

        self._mock_state = {
            'location': 'countertop 1',
            'inventory': [],
            'task_type': extra_info.get('task_type', 'pick_and_place'),
            'target_object': extra_info.get('target_object', 'apple'),
            'target_receptacle': extra_info.get('target_receptacle', 'countertop 1'),
            'object_location': extra_info.get('object_location', 'fridge 1'),
            'object_state': 'dirty',
            'task_complete': False,
            'containers_opened': set(),
        }

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
            if self._use_real:
                return self._step_real(command)
            else:
                return self._step_mock(command)

        return self._obs_with_commands(f'Unknown tool "{func}". Use "action" tool.')

    def _step_real(self, command: str) -> dict:
        """Execute action in real TextWorld environment."""
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

            # Check win
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
            print(f'[ALF REAL] step={self._step_count} cmd="{command[:60]}" won={self._won}')
            return self._obs_with_commands(obs_text)

        except Exception as e:
            print(f'[ALF REAL] Error: {e}')
            return self._obs_with_commands(f'Error executing action: {e}')

    def _step_mock(self, command: str) -> dict:
        """Execute action in mock state machine."""
        s = self._mock_state
        cmd = command.lower().strip()

        ALL_RECEPTACLES = [
            'countertop 1', 'countertop 2', 'cabinet 1', 'cabinet 2',
            'drawer 1', 'fridge 1', 'sink 1', 'sinkbasin 1',
            'stoveburner 1', 'microwave 1', 'garbagecan 1',
            'coffeetable 1', 'sofa 1', 'shelf 1', 'shelf 2',
            'armchair 1', 'sidetable 1',
            'bed 1', 'dresser 1', 'desk 1', 'drawer 2',
            'sidetable 2', 'shelf 3',
            'sinkbasin 2', 'toilet 1', 'bathtub 1', 'shelf 4', 'cabinet 3',
        ]

        if cmd.startswith('go to '):
            target = cmd[6:].strip()
            match = next((r for r in ALL_RECEPTACLES if target in r or r in target), None)
            if match:
                s['location'] = match
                obs = f'You arrive at {match}.'
                if match == s['object_location'] and s['target_object'] not in s['inventory']:
                    if any(w in match for w in ['fridge', 'cabinet', 'drawer', 'microwave']) and match not in s['containers_opened']:
                        obs += f' The {match} is closed.'
                    else:
                        obs += f' You see a {s["target_object"]} here.'
                return self._obs_with_commands(obs)
            return self._obs_with_commands(f"Can't find {target}.")

        if cmd.startswith('take '):
            if s['target_object'] in cmd and s['location'] == s['object_location']:
                if any(w in s['location'] for w in ['fridge', 'cabinet', 'drawer', 'microwave']) and s['location'] not in s['containers_opened']:
                    return self._obs_with_commands(f'The {s["location"]} is closed.')
                s['inventory'].append(s['target_object'])
                return self._obs_with_commands(f'You pick up the {s["target_object"]}.')
            return self._obs_with_commands("You can't take that.")

        if cmd.startswith('put '):
            if s['target_object'] in s['inventory'] and s['target_object'] in cmd:
                at_target = s['target_receptacle'] in s['location'] or s['location'] in s['target_receptacle'] or s['target_receptacle'] in cmd
                if at_target:
                    if 'clean' in s['task_type'] and s['object_state'] != 'clean':
                        return self._obs_with_commands(f'The {s["target_object"]} needs cleaning.')
                    if 'heat' in s['task_type'] and s['object_state'] != 'hot':
                        return self._obs_with_commands(f'The {s["target_object"]} needs heating.')
                    if 'cool' in s['task_type'] and s['object_state'] != 'cool':
                        return self._obs_with_commands(f'The {s["target_object"]} needs cooling.')
                    s['inventory'].remove(s['target_object'])
                    s['task_complete'] = True
                    self.is_finish = True
                    self.finish = True
                    return {'observation': f'You put the {s["target_object"]} on {s["target_receptacle"]}. Task complete!'}
                return self._obs_with_commands(f'Go to {s["target_receptacle"]} first.')
            return self._obs_with_commands("Can't put that here.")

        if cmd.startswith('clean '):
            if s['target_object'] in s['inventory'] and ('sinkbasin' in s['location'] or 'sink' in s['location']):
                s['object_state'] = 'clean'
                return self._obs_with_commands(f'You clean the {s["target_object"]}.')
            return self._obs_with_commands("Need sink + holding object.")

        if cmd.startswith('heat '):
            if s['target_object'] in s['inventory'] and 'microwave' in s['location']:
                s['object_state'] = 'hot'
                return self._obs_with_commands(f'You heat the {s["target_object"]}.')
            return self._obs_with_commands("Need microwave + holding object.")

        if cmd.startswith('cool '):
            if s['target_object'] in s['inventory'] and 'fridge' in s['location']:
                s['object_state'] = 'cool'
                return self._obs_with_commands(f'You cool the {s["target_object"]}.')
            return self._obs_with_commands("Need fridge + holding object.")

        if cmd.startswith('open '):
            target = cmd[5:].strip()
            match = next((r for r in ALL_RECEPTACLES if target in r or r in target), s['location'])
            s['containers_opened'].add(match)
            obs = f'You open the {match}.'
            if match == s['object_location'] and s['target_object'] not in s['inventory']:
                obs += f' You see a {s["target_object"]} inside.'
            return self._obs_with_commands(obs)

        if cmd.startswith('close '):
            target = cmd[6:].strip()
            match = next((r for r in ALL_RECEPTACLES if target in r or r in target), s['location'])
            s['containers_opened'].discard(match)
            return self._obs_with_commands(f'You close the {match}.')

        if cmd in ('look', 'inventory') or cmd.startswith('examine'):
            obs = f'At {s["location"]}.'
            if s['inventory']:
                obs += f' Carrying: {", ".join(s["inventory"])}.'
            if s['location'] == s['object_location'] and s['target_object'] not in s['inventory']:
                obs += f' See: {s["target_object"]}.'
            return self._obs_with_commands(obs)

        return self._obs_with_commands(f'Unknown: "{command}".')

    def _obs_with_commands(self, obs_text: str) -> dict:
        """Return observation without admissible commands (MemexRL style)."""
        return {'observation': obs_text}

    def _get_mock_admissible(self) -> list:
        """Generate mock admissible commands."""
        if self._mock_state is None:
            return ['look']
        s = self._mock_state
        cmds = ['look', 'inventory']
        ALL_RECEPTACLES = [
            'countertop 1', 'countertop 2', 'cabinet 1', 'cabinet 2',
            'drawer 1', 'fridge 1', 'sink 1', 'sinkbasin 1',
            'stoveburner 1', 'microwave 1', 'garbagecan 1',
            'coffeetable 1', 'sofa 1', 'shelf 1', 'shelf 2', 'armchair 1', 'sidetable 1',
            'bed 1', 'dresser 1', 'desk 1', 'drawer 2', 'sidetable 2', 'shelf 3',
            'sinkbasin 2', 'toilet 1', 'bathtub 1', 'shelf 4', 'cabinet 3',
        ]
        for r in ALL_RECEPTACLES:
            if r != s['location']:
                cmds.append(f'go to {r}')
        if s['location'] == s['object_location'] and s['target_object'] not in s['inventory']:
            cmds.append(f'take {s["target_object"]} from {s["location"]}')
        if s['target_object'] in s['inventory']:
            cmds.append(f'put {s["target_object"]} in/on {s["location"]}')
        return cmds[:20]

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
        if self._use_real:
            reward = 1.0 if self._won else 0.0
        else:
            reward = 1.0 if (self._mock_state and self._mock_state.get('task_complete', False)) else 0.0
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
