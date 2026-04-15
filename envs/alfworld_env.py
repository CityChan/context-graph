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
        # "hard" in ability → MemexRL-style: hide admissible commands
        self._hide_admissible = "hard" in ability.lower()

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
        """Initialize mock state machine (multi-object, possibly across multiple containers)."""
        self.instance_info['problem_statement'] = extra_info.get('task_desc', '')

        target_objects = list(extra_info.get('target_objects', []))
        target_receptacles = list(extra_info.get('target_receptacles', []))
        # Backward compat: if old single-object fields present, promote to lists
        if not target_objects and extra_info.get('target_object'):
            target_objects = [extra_info['target_object']]
            target_receptacles = [extra_info.get('target_receptacle', 'countertop 1')]

        # Map each object to its target receptacle (parallel lists)
        object_to_receptacle = dict(zip(target_objects, target_receptacles))

        # Map each object to its source container.
        # Prefer explicit per-object map; fall back to single shared container for all.
        object_location_map = dict(extra_info.get('object_location_map', {}) or {})
        default_loc = extra_info.get('object_location', 'fridge 1')
        for o in target_objects:
            object_location_map.setdefault(o, default_loc)

        self._mock_state = {
            'location': 'countertop 1',
            'inventory': [],
            'task_type': extra_info.get('task_type', 'pick_and_place_multi'),
            'target_objects': target_objects,
            'target_receptacles': target_receptacles,
            'object_to_receptacle': object_to_receptacle,
            'object_location_map': object_location_map,
            'objects_placed': set(),
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

        remaining = [o for o in s['target_objects'] if o not in s['objects_placed']]
        loc_map = s['object_location_map']
        # Objects currently present at a given receptacle (not yet picked up or placed)
        def objects_at(recep):
            return [o for o in remaining if loc_map.get(o) == recep and o not in s['inventory']]

        if cmd.startswith('go to '):
            target = cmd[6:].strip()
            match = next((r for r in ALL_RECEPTACLES if target in r or r in target), None)
            if match:
                s['location'] = match
                obs = f'You arrive at {match}.'
                visible = objects_at(match)
                if visible:
                    if any(w in match for w in ['fridge', 'cabinet', 'drawer', 'microwave']) and match not in s['containers_opened']:
                        obs += f' The {match} is closed.'
                    else:
                        obs += f' You see: {", ".join(visible)}.'
                return self._obs_with_commands(obs)
            return self._obs_with_commands(f"Can't find {target}.")

        if cmd.startswith('take '):
            # Match any target object present in command; must be at that object's container
            obj = next((o for o in s['target_objects'] if o in cmd), None)
            if obj and s['location'] == loc_map.get(obj) and obj not in s['objects_placed'] and obj not in s['inventory']:
                if any(w in s['location'] for w in ['fridge', 'cabinet', 'drawer', 'microwave']) and s['location'] not in s['containers_opened']:
                    return self._obs_with_commands(f'The {s["location"]} is closed.')
                s['inventory'].append(obj)
                return self._obs_with_commands(f'You pick up the {obj}.')
            return self._obs_with_commands("You can't take that.")

        if cmd.startswith('put '):
            # Find which inventory object the agent is trying to put
            obj = next((o for o in s['inventory'] if o in cmd), None)
            if obj is None:
                return self._obs_with_commands("You aren't carrying that.")
            target_recep = s['object_to_receptacle'].get(obj, '')
            at_target = target_recep and (target_recep in s['location'] or s['location'] in target_recep or target_recep in cmd)
            if at_target:
                s['inventory'].remove(obj)
                s['objects_placed'].add(obj)
                if len(s['objects_placed']) == len(s['target_objects']):
                    s['task_complete'] = True
                    self.is_finish = True
                    self.finish = True
                    return {'observation': f'You put the {obj} on {target_recep}. Task complete!'}
                return self._obs_with_commands(
                    f'You put the {obj} on {target_recep}. ({len(s["objects_placed"])}/{len(s["target_objects"])} done)'
                )
            return self._obs_with_commands(f'Go to {target_recep} first to put the {obj}.')

        if cmd.startswith('open '):
            target = cmd[5:].strip()
            match = next((r for r in ALL_RECEPTACLES if target in r or r in target), s['location'])
            s['containers_opened'].add(match)
            obs = f'You open the {match}.'
            visible = objects_at(match)
            if visible:
                obs += f' You see inside: {", ".join(visible)}.'
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
            visible = objects_at(s['location'])
            if visible:
                obs += f' See here: {", ".join(visible)}.'
            return self._obs_with_commands(obs)

        return self._obs_with_commands(f'Unknown: "{command}".')

    def _obs_with_commands(self, obs_text: str) -> dict:
        """Return observation with admissible commands (unless hidden in hard mode)."""
        if self._hide_admissible:
            return {'observation': obs_text}
        cmds = self._admissible_commands if self._use_real else self._get_mock_admissible()
        if cmds:
            cmd_str = ", ".join(cmds)
            obs_text += f"\n\nAdmissible commands: [{cmd_str}]"
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
        remaining = [o for o in s['target_objects'] if o not in s['objects_placed']]
        loc_map = s.get('object_location_map', {})
        # take any remaining object that's at current location
        for o in remaining:
            if o not in s['inventory'] and loc_map.get(o) == s['location']:
                cmds.append(f'take {o} from {s["location"]}')
        # put inventory objects
        for o in s['inventory']:
            cmds.append(f'put {o} in/on {s["location"]}')
        # open container if closed
        if any(w in s['location'] for w in ['fridge', 'cabinet', 'drawer', 'microwave']) and s['location'] not in s['containers_opened']:
            cmds.append(f'open {s["location"]}')
        return cmds[:25]

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
