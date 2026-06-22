"""ReAct agent for code-execution tasks (ScienceAgentBench).

Single-threaded ReAct loop. Differs from agents/react_agent.py ONLY in:
  - the system prompt is built via prompts_code.create_chat_code (which
    advertises the `python_exec` tool instead of `search`)
  - the workflow default is 'code' instead of 'search'

Env dispatch is unchanged: select_env() reads `ability` from the task
and returns ScienceAgentEnv for `ScienceAgentBench` ability. The env
handles tool-call parsing for python_exec + finish.
"""

import os
import asyncio
import copy
from uuid import uuid4

from verl import DataProto
from .utils import Agent, select_env, TaskContext, run_action, AgentLoopOutput, AgentLoopMetrics
from .prompts_code import create_chat_code


async def process_item(
        item: DataProto,
        context: TaskContext,
) -> AgentLoopOutput:
    os.environ["no_proxy"] = ""
    tokenizer = context.tokenizer
    config = context.config.actor_rollout_ref.rollout
    is_train = context.is_train

    if not is_train:
        if getattr(config.plugin, "val_response_length", None):
            config.response_length = getattr(config.plugin, "val_response_length", None)

    # Helper: verl may squeeze the batch dim at val time, yielding 0-d arrays.
    def _get(arr):
        import numpy as np
        v = np.asarray(arr)
        return v.item() if v.ndim == 0 else v[0]

    ability = _get(item.non_tensor_batch['ability'])

    uid = item.non_tensor_batch.get('uid', uuid4().hex)
    gen_uid = item.non_tensor_batch.get('gen_uid', None)
    EnvClass = select_env(ability, config)
    print(is_train, EnvClass)
    env = EnvClass(config, tokenizer, ability)

    try:
        await env.init_env(item)
    except Exception as e:
        print(f"[Error] during environment init: {str(e)}")

    workflow = _get(item.non_tensor_batch['extra_info']).get('workflow', None) or getattr(
        config.plugin, "workflow", "code")
    user_prompt = create_chat_code(env.instance_info['problem_statement'], workflow, item, env=env)
    max_turn = getattr(config.plugin, 'max_turn', 32) if config.plugin else 32

    llm_client = context.llm_client
    prompt_turn = len(user_prompt)

    agent = Agent(llm_client, user_prompt, tokenizer, config, prompt_turn=prompt_turn)
    iteration = 0
    natural_finish = False
    while iteration < max_turn:
        iteration += 1
        response = await agent.step()
        if response is None:
            natural_finish = True
            break
        observation = await run_action(env, response)
        if observation is None:
            natural_finish = True
            break
        agent.append({'role': 'user', 'content': observation})

    mask_rollout = (iteration >= max_turn) and not natural_finish

    print('[TASK] Task Finish, Start Reward')
    try:
        score_msg, reward, reward_dict = await asyncio.wait_for(
            env.get_reward(item, agent.messages(), context), timeout=60 * 10)
        score = (score_msg, reward)
        print(score)
    except Exception as e:
        print(f"[Error] Getting reward: {e}")
        score, reward_dict = ("", 0), {"ans_reward": 0.0, "format_reward": 0.0, "ref_reward": 0.0}

    if not hasattr(env, 'stats') or env.stats is None:
        env.stats = {}
    env.stats['task_reward'] = float(score[1])
    env.stats['main_turn'] = int(iteration)
    env.stats['is_branch'] = 0
    env.stats['branch_success'] = 0
    # Surface real-eval VER (valid_execution) + produced-file count so they
    # aggregate into val/* metrics. At 8B zero-shot SR floors to 0, so VER is
    # the signal that separates the agents. Only present under SAB_REAL_EVAL=1.
    if isinstance(reward_dict, dict):
        for _k in ('valid_execution', 'produced_files'):
            if _k in reward_dict:
                try:
                    env.stats[_k] = float(reward_dict[_k])
                except (TypeError, ValueError):
                    pass

    # Clean up the sandbox (release the namespace dict so GC can reclaim it).
    try:
        if hasattr(env, 'close'):
            env.close()
    except Exception:
        pass

    out_data = await agent.get_data()
    agent_reward = score[1]
    out = AgentLoopOutput(
        prompt_ids=out_data['prompt_ids'],
        response_ids=out_data['response_ids'],
        response_mask=out_data['response_mask'],
        response_logprobs=out_data['response_logprobs'],
        multi_modal_data={},
        metrics=AgentLoopMetrics(),
        reward_score=agent_reward,
        num_turns=out_data['num_turns'],
        extra_fields={
            'messages': out_data['messages'],
            'env_stats': copy.deepcopy(env.stats),
            'mask_rollout': mask_rollout,
            'is_finish': natural_finish,
            'process_reward_mask': out_data['process_reward_mask'],
            'uid': uid,
            'gen_uid': gen_uid,
        },
    )
    return out
