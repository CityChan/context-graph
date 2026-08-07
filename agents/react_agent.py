import os
import asyncio
import copy
from uuid import uuid4

from verl import DataProto
from .utils import Agent, select_env, TaskContext, run_action, AgentLoopOutput, AgentLoopMetrics
from .prompts import create_chat


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
    # Bare [0] indexing crashes on those; .item() / [0] dispatch handles both
    # shapes. Mirrors agents/fold_agent.py:70-73 and graph_agent_isolated.py:223-226.
    def _get(arr):
        import numpy as np
        v = np.asarray(arr)
        return v.item() if v.ndim == 0 else v[0]

    ability = _get(item.non_tensor_batch['ability'])

    uid = item.non_tensor_batch.get('uid', uuid4().hex)
    gen_uid = item.non_tensor_batch.get('gen_uid', None)
    # Select env
    EnvClass = select_env(ability, config, )
    print(is_train, EnvClass)
    env = EnvClass(config, tokenizer, ability)

    try:
        await env.init_env(item)
    except Exception as e:
        print(f"[Error] during environment init: {str(e)}")

    workflow = _get(item.non_tensor_batch['extra_info']).get('workflow', None) or getattr(config.plugin, "workflow",
                                                                                          "search")
    user_prompt = create_chat(env.instance_info['problem_statement'], workflow, item)
    max_turn = getattr(config.plugin, 'max_turn', 64) if config.plugin else 64

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

    # Only the environment can establish that the task actually finished.
    # A None completion commonly means the token/context budget was exhausted;
    # treating that as a successful finish made GAIA finished_items misleading.
    is_finish = bool(getattr(env, 'is_finish', False) or getattr(env, 'finish', False))

    print('[TASK] Task Finish, Start Reward')
    try:
        score_msg, reward, reward_dict = await asyncio.wait_for(
            env.get_reward(item, agent.messages(), context), timeout=60 * 10)
        score = (score_msg, reward)
        print(score)
    except Exception as e:
        print(f"[Error] Getting reward: {e}")
        score, reward_dict = ("", 0), {"ans_reward": 0.0, "format_reward": 0.0, "ref_reward": 0.0}

    mask_rollout = not (is_finish or score[1] > 0)

    # Populate env.stats with the minimum keys the reward manager lifts so
    # wandb shows reward/task_reward / reward/main_turn / reward/avg_num_turns
    # in line with the fold/ctxgraph runs.
    if not hasattr(env, 'stats') or env.stats is None:
        env.stats = {}
    env.stats['task_reward'] = float(score[1])
    env.stats['main_turn'] = int(iteration)
    env.stats['is_branch'] = 0
    env.stats['branch_success'] = 0

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
            'is_finish': is_finish,
            'process_reward_mask': out_data['process_reward_mask'],
            'uid': uid,
            'gen_uid': gen_uid,
        },
    )
    return out


# @register_handler("agent/react_agent")
# class ReActAgent(AsyncAgent):
#     async def __call__(self, item: DataProto, context: TaskContext, **kwargs):
#         return await process_single_batch(item, context)
