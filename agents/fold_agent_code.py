import os
import time
import copy
import asyncio
from .environment_lifecycle import managed_environment
from .session_summary import iter_agent_data
from functools import partial
import random
from uuid import uuid4
from typing import Any, Union

# Fold agent for code-execution tasks (ScienceAgentBench).
#
# This file is a strict fork of fold_agent.py with the minimum changes needed
# to route through the code-domain prompt builder. The fold-grpo session
# management, branching, reward shaping, and process_reward handling are
# byte-for-byte identical so paper-grade comparison between BC-Plus fold and
# ScienceAgentBench fold tests purely the agent-loop substrate, not loop
# semantics.
#
# Edits vs fold_agent.py (5 lines total):
#   - import block: drop create_chat / *_SEARCH; add create_chat_code
#   - default workflow: "search" -> "code_branch"
#   - branch_prompt: drop search-vs-code dispatch (always BRANCH_MESSAGE)
#   - prompt builder call: create_chat(...) -> create_chat_code(...)
#   - env.close() on completion (sandbox cleanup)

from verl import DataProto
from .utils import Agent, select_env, truncate_text, is_weird, TaskContext, run_action, AgentLoopOutput, AgentLoopMetrics
from .rollout_status import classify_rollout_status, validate_session_summary
from .prompts import BRANCH_MESSAGE
from .prompts_code import create_chat_code
from .verifier import judge_scope


from .utils import print_chat

from .agent_text import extract_fn_call, extract_summary, clean_response


async def process_item(
        item: DataProto,
        context: TaskContext,
) -> Union[AgentLoopOutput, list[AgentLoopOutput]]:
    os.environ["no_proxy"] = ""
    tokenizer = context.tokenizer

    config = context.config.actor_rollout_ref.rollout
    validate_session_summary(config.plugin)
    is_train = context.is_train

    if not is_train:
        if getattr(config.plugin, "val_response_length", None):
            config.response_length = getattr(config.plugin, "val_response_length", None)

    # Helper: verl may squeeze batch dim, yielding 0-d arrays
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

    async with managed_environment(env):
        try:
            await env.init_env(item)
        except Exception as e:
            print(f"[Error] during environment init: {str(e)}")
            raise

        # Create prompt
        workflow = _get(item.non_tensor_batch['extra_info']).get('workflow', None) or getattr(config.plugin, "workflow", "code_branch")
        user_prompt = create_chat_code(env.instance_info['problem_statement'], workflow, item, env=env)

        branch_prompt = BRANCH_MESSAGE

        max_turn = getattr(config.plugin, 'max_turn', 64) if config.plugin else 64
        max_session = getattr(config.plugin, "max_session", 5)
        if not is_train:
            max_session = getattr(config.plugin, "val_max_session", max_session)
        session_timeout = getattr(config.plugin, "session_timeout", 90 * 60)
        process_reward = getattr(config.plugin, "process_reward", None)
        if process_reward is not None and isinstance(process_reward, str) and process_reward.lower() == "none":
            process_reward = None
        max_traj = getattr(config.plugin, "max_traj", None)

        llm_client = context.llm_client

        prompt_turn = len(user_prompt)
        agent = dict()
        agent['main'] = Agent(llm_client, user_prompt, tokenizer, config, prompt_turn=prompt_turn)
        branches = []
        branch_tasks = {}
        branch_return = {}
        init_len = len(agent['main'].context())
        session_start_time = time.time()
        iteration = 0
        mask_rollout = True  # If True then no grad update on this traj
        timed_out = False
        session_message = []
        while iteration < max_turn:
            if time.time() - session_start_time > session_timeout:
                print('[SESSION] Session Timeout')
                timed_out = True
                break

            summary_start = len(agent['main'].messages())
            await agent['main'].maybe_restart_session(0)
            session_message.extend(agent['main'].messages()[summary_start:])

            iteration += 1


            response = await agent['main'].step()
            # print(response)

            if response is None:
                break

            session_message.append({'role': 'assistant', 'content': response})
            fn_call = extract_fn_call(response)
            if fn_call is not None and fn_call['function'] == 'branch':
                if len(branches) + 1 > max_session:
                    observation = f"You've already reached the limit of {len(branches)} branch calls. Continue working independently."
                else:
                    description = fn_call['arguments'].get('description', 'Agent')
                    message_to_branch = fn_call['arguments'].get('prompt', 'Empty prompt')
                    print('[BRANCH]', description, len(agent['main'].context()))
                    # print(message_to_branch)
                    agent_name = f"#{len(branches)}-" + description.replace(' ', '_')
                    branches.append(agent_name)
                    branch_tasks[agent_name] = message_to_branch
                    history = agent['main'].working_messages()
                    agent[agent_name] = Agent(llm_client, history, tokenizer, config, prompt_turn=prompt_turn)
                    branch_prompt_formatted = branch_prompt.format(message=message_to_branch)
                    agent[agent_name].append({'role': 'user', 'content': branch_prompt_formatted})
                    agent_return = await agent[agent_name].react(
                        partial(run_action, env),
                        max_turn=max_turn,
                        max_tokens=getattr(config.plugin, "branch_len", None),
                        session_timeout=session_timeout - time.time() + session_start_time,
                        should_continue=lambda resp: '<function=return>' not in resp,
                        safe_finish=lambda
                            x: "You are in branch mode and cannot branch task or finish the task. Use the `return` tool to go back to the main agent." if '<function=finish>' in x or '<function=branch>' in x else None,
                        summary_prompt="The context limit has been exceeded for the branch. Please finish the sub task directly and clearly state the progress made and the pending jobs of the sub task. Only summarize the sub task progress, using the return tool.",
                        observation_prompt=f"* You are now in branch mode: {description}. Conduct the sub task based on instruction, and when you complete the assigned sub task, use return tool to return, do not perform action beyond the assigned sub task.",
                    )
                    iteration += agent_return['iteration']
                    last_response = agent_return['last_response']
                    session_message.extend(agent[agent_name].messages()[len(history):])
                    fn_call = extract_fn_call(last_response)
                    branch_message = None
                    if fn_call is not None and fn_call['function'] == 'return':
                        if 'message' in fn_call['arguments']:
                            branch_message = fn_call['arguments'].get('message', 'Empty message')
                            branch_message = f'Branch has finished its task, the returned message is:\n\n{branch_message}'
                    elif fn_call is not None and fn_call['function'] == 'finish':
                        if 'message' in fn_call['arguments']:
                            branch_message = fn_call['arguments'].get('message', 'Empty message')
                            branch_message = f'Branch has finished its task, the returned message is:\n\n{branch_message}'
                    if branch_message is None:
                        branch_message = f'Branch has finished its task. The last message was:\n\n{clean_response(last_response)}'
                    observation = branch_message
                    branch_return[agent_name] = observation
                    # print(observation)
            else:
                observation = await run_action(env, response)
                if observation is None:
                    mask_rollout = False
                    break

            if agent['main'].chat[-1]['role'] == 'user':
                print('[ROLE ERROR]')
                print(agent['main'].chat[-1])
                agent['main'].append({'role': 'assistant', 'content': str(response)})

            if process_reward:
                observation = truncate_text(observation, max_lines=100, merge_repeat=True, merge_num=4)
            # print(observation)
            agent['main'].append({'role': 'user', 'content': observation})
            session_message.append({'role': 'user', 'content': observation})

        env.stats['session_time'] = time.time() - session_start_time

        print('[TASK] Task Finish, Start Reward')
        try:
            score_msg, reward, reward_dict = await asyncio.wait_for(
                env.get_reward(item, agent['main'].messages(), context), timeout=60 * 10)
            score = (score_msg, reward)
            print(score)
        except Exception as e:
            print(f"[Error] Getting reward: {e}")
            score, reward_dict = ("", 0), {"ans_reward": 0.0, "format_reward": 0.0, "ref_reward": 0.0}

        outs = []
        env.stats['get_final_score'] = score[1]
        # Mirror the field CtxGraph logs so wandb has reward/task_reward on
        # both runs for apples-to-apples task-accuracy comparison.
        env.stats['task_reward'] = float(score[1])
        # Surface real-eval VER (valid_execution) + produced-file count so they
        # aggregate into val/* metrics. At 8B zero-shot SR floors to 0, so VER is
        # the signal that separates the agents. Only present under SAB_REAL_EVAL=1.
        if isinstance(reward_dict, dict):
            for _k in (
                'valid_execution', 'produced_files', 'valid_result_json',
                'hms_score', 'hms_context_recall', 'hms_mean_accuracy', 'judge_error',
            ):
                if _k in reward_dict:
                    try:
                        env.stats[_k] = float(reward_dict[_k])
                    except (TypeError, ValueError):
                        pass
        env.stats['summary_restarts'] = len(getattr(agent['main'], 'summary_sessions', []))
        env.stats['traj_num'] = len(agent)
        main_response_tokens = agent['main'].response_tokens_used()
        main_context_tokens = len(agent['main'].context())
        working_context_limit = config.prompt_length + config.response_length
        env.stats['main_len'] = min(main_response_tokens, config.response_length)
        env.stats['main_context_tokens'] = main_context_tokens
        env.stats['working_context_limit'] = working_context_limit
        env.stats['total_token'] = len(tokenizer.encode(print_chat(user_prompt + session_message)))
        env.stats['main_turn'] = len(agent['main'].messages())
        env.stats['is_branch'] = int(len(agent) > 1)
        env.stats['branch_success'] = int(int(len(agent) > 1) * score[1])
        env.stats['use_all_branch'] = int(len(branches) + 1 > max_session)

        if getattr(env, 'is_finish', False) or getattr(env, 'finish', False):
            mask_rollout = False
        if score[1] > 0:
            mask_rollout = False

        is_finish = getattr(env, 'is_finish', False) or getattr(env, 'finish', False)
        rollout_status = classify_rollout_status(
            response_tokens=main_response_tokens,
            response_limit=config.response_length,
            main_context_tokens=main_context_tokens,
            working_context_limit=working_context_limit,
            is_finish=is_finish,
            iteration=iteration,
            max_turn=max_turn,
            timed_out=timed_out,
        )
        env.stats.update({k: v for k, v in rollout_status.items() if k != 'termination_reason'})
        env.stats['concise_main'] = 1 - rollout_status['unfolded_main']
        if getattr(config.plugin, "must_finish", None):
            if not is_finish:
                score = ('', 0)

        if process_reward and is_train:
            mask_rollout = False
            env.stats['concise_main'] = int(len(agent['main'].context()) - init_len <= config.response_length * 0.5)
            if 'cjk' in process_reward:
                env.stats['is_cjk'] = 0
                for name in agent:
                    for i, turn in enumerate(agent[name].chat):
                        if is_weird(str(turn)):
                            print('[CJK ERROR]')
                            print(str(turn))
                            env.stats['is_cjk'] = 1
                            agent[name].set_process_reward(i, -1)
                            if 'flat' in process_reward:
                                agent[name].set_cache('reward', 0)
            if score[1] > 0:
                # Check main
                if len(agent['main'].context()) - init_len > config.response_length * 0.5:
                    bad_turn = [i for i, turn in enumerate(agent['main'].messages()) if
                                '<function=branch>' not in str(turn) and '<function=finish>' not in str(turn)]
                    agent['main'].set_process_reward(bad_turn, -1)

                if len(agent) == 1:
                    agent['main'].set_process_reward('all', -1)
                    if 'flat' in process_reward:
                        agent['main'].set_cache('reward', 1 - 1)

                # Scope check
                if 'scope' in process_reward:
                    env.stats['scope_judge'] = 1
                    for name in branches:
                        assigned_task = branch_tasks[name]
                        return_message = branch_return[name]
                        is_focus, justification = await judge_scope(assigned_task, return_message)
                        if is_focus < 0:  # scope check, skip summary turn
                            print(f'[FOCUS] Branch beyond focus: //{name}//. {justification}')
                            agent[name].set_process_reward([i for i in range(len(agent[name].chat) - 1)], -0.2)
                            if 'flat' in process_reward:
                                agent[name].set_cache('reward', 1 - 0.2)
                            env.stats['scope_judge'] = 0
                        elif is_focus > 0:
                            agent[name].set_process_reward([i for i in range(len(agent[name].chat) - 1)], 0.2)
                            if 'flat' in process_reward:
                                agent[name].set_cache('reward', 1 + 0.2)
                # Tool call error
                for name in branches:
                    for i, turn in enumerate(agent[name].chat):
                        ERR_MARKERS = (
                            'Failed to validate tool call',
                            'Failed to parse tool call',
                            'You are in branch mode and cannot branch task or finish the task.',
                            'No function call was detected in the model response',
                            '[Error] The "search" function requires a "query" argument',
                            '[Error] The "open_page" function requires either a "docid" or a "url".',
                            '[Error] The function',
                        )
                        if any(m in str(turn) for m in ERR_MARKERS):
                            agent[name].set_process_reward(i - 1, -1)
            else:
                is_finish = getattr(env, 'is_finish', False) or getattr(env, 'finish', False)
                if 'drop_fail' in process_reward:
                    if not is_finish:
                        for name in branches:
                            if 'cjk' in process_reward:
                                should_drop = True
                                for i, turn in enumerate(agent[name].chat):
                                    if is_weird(str(turn)):
                                        agent[name].set_process_reward(i, -2)
                                        if 'flat' in process_reward:
                                            agent[name].set_cache('reward', -1)
                                        should_drop = False
                                if should_drop:
                                    agent.pop(name)
                            else:
                                agent.pop(name)  # drop all branch if not finish (overlong mask)
                # Scope check + reward
                if 'reward_scope' in process_reward:
                    env.stats['scope_judge'] = 1
                    for name in branches:
                        assigned_task = branch_tasks[name]
                        return_message = branch_return[name]
                        is_focus, justification = await judge_scope(assigned_task, return_message)
                        if is_focus < 0:  # scope check, skip summary turn
                            print(f'[FOCUS] Branch beyond focus: //{name}//. {justification}')
                            agent[name].set_process_reward([i for i in range(len(agent[name].chat) - 1)], -0.2)
                            if 'flat' in process_reward:
                                agent[name].set_cache('reward', 0 - 0.2)
                            env.stats['scope_judge'] = 0
                        elif is_focus > 0:
                            agent[name].set_process_reward([i for i in range(len(agent[name].chat) - 1)], 0.2)
                            if 'flat' in process_reward:
                                agent[name].set_cache('reward', 0 + 0.2)

        async for name, out in iter_agent_data(agent, is_train, max_traj):
            agent_reward = score[1]
            if process_reward is not None and 'flat' in process_reward and 'reward' in agent[name].info_cache:
                agent_reward = agent[name].info_cache['reward']

            # Always pass process_reward_mask (verl requires the key to exist).
            # If the mask is all zeros, verl's compute_foldgrpo_advantage will
            # detect this and skip the gmin/gmax patching path.
            out = AgentLoopOutput(
                prompt_ids=out['prompt_ids'],
                response_ids=out['response_ids'],
                response_mask=out['response_mask'],
                response_logprobs=out['response_logprobs'],
                multi_modal_data={},
                reward_score=agent_reward,
                num_turns=out['num_turns'],
                metrics=AgentLoopMetrics(),
                extra_fields={
                    'messages': out['messages'],
                    'env_stats': copy.deepcopy(env.stats) if hasattr(env, 'stats') else {},
                    'num_branches': len(branches),
                    'branch_names': branches,
                    'mask_rollout': mask_rollout,
                    'overlong': rollout_status['overlong'],
                    'no_finish': rollout_status['no_finish'],
                    'hit_token_limit': rollout_status['hit_token_limit'],
                    'hit_max_turn': rollout_status['hit_max_turn'],
                    'hit_timeout': rollout_status['hit_timeout'],
                    'unfolded_main': rollout_status['unfolded_main'],
                    'termination_reason': rollout_status['termination_reason'],
                    'is_finish': is_finish,
                    'agent_name': name,
                    'summary_session_index': out.get('summary_session_index', 0),
                    'message_str': print_chat(session_message),
                    'meta_info': f"N: {len(agent)} | {name}",
                    'process_reward_mask': out['process_reward_mask'],
                    'uid': uid,
                    'gen_uid': gen_uid,
                }
            )

            outs.append(copy.deepcopy(out))


        return outs
