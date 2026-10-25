"""Opt-in inference recovery for BC-P's native XML task protocol."""
import copy
import re


def has_task_call(text, role):
    visible = text.rsplit('</think>', 1)[-1]
    if '<think>' in visible:
        return False
    required = {'search': ('query',), 'branch': ('description', 'prompt'),
                'finish': ('answer',), 'return': ('message',)}
    allowed = {'search', 'open_page', 'branch', 'finish'} if role == 'main' else {'search', 'open_page', 'return'}
    for name, body in re.findall(r'<function=([^>]+)>(.*?)</function>', visible, re.S):
        if name not in allowed:
            continue
        params = dict(re.findall(r'<parameter=([^>]+)>(.*?)</parameter>', body, re.S))
        if name == 'open_page':
            if params.get('docid', '').strip() or params.get('url', '').strip():
                return True
        elif all(params.get(k, '').strip() for k in required[name]):
            return True
    return False


def constraint(role, final=False):
    if role not in ('main', 'branch'):
        raise ValueError('Unknown task recovery role')
    ws = r'[ \t\r\n]*'

    def param(name):
        # Bound the request in tokens, not each argument in characters. A hard
        # character ceiling forces XML closure mid-sentence while still passing
        # syntactic validation (notably branch prompts and evidence returns).
        return '<parameter=' + name + '>' + r'[^<>\s][^<>]*</parameter>' + ws

    def call(name, body):
        return '<function=' + name + '>' + ws + body + '</function>'

    ending = call('finish', param('answer') + '(?:' + param('explanation') + ')?' + '(?:' + param('confidence') + ')?') if role == 'main' else call('return', param('message'))
    calls = [ending]
    if not final:
        calls += [call('search', param('query')), call('open_page', '(?:' + param('docid') + '|' + param('url') + ')')]
        if role == 'main':
            calls.append(call('branch', param('description') + param('prompt')))
    return {'regex': ws + '(?:' + '|'.join(calls) + ')' + ws}


async def recover(agent, max_new_tokens=None):
    """One request in the ordinary turn slot; never recursively retry or dispatch."""
    from .utils import _apply_chat_template
    role = agent.task_recovery_role
    final = bool(getattr(agent, 'force_task_finish', False))
    schema = constraint(role, final)
    messages = copy.deepcopy(agent.working_messages())
    instruction = ('Submit exactly one finish tool call using available evidence.' if final else
                   'The last response did not execute a task tool. Output exactly one complete native XML tool call now. '
                   'Choose the next task action yourself; do not repeat analysis. Main may branch; branches must return instead of finish.')
    instruction += (' Keep arguments concise and complete. Preserve the essential task constraints in branch prompts, '
                    'and distinguish verified evidence from hypotheses in returns. Reserve space to close every XML tag.')
    if messages and messages[-1]['role'] == 'user':
        messages[-1]['content'] += '\n\n' + instruction
    else:
        messages.append({'role': 'user', 'content': instruction})
    template = dict(agent.chat_template_kwargs, enable_thinking=False)
    base = _apply_chat_template(agent.tokenizer, messages, agent.config, tokenize=True, add_generation_prompt=False, **template)
    prompt = _apply_chat_template(agent.tokenizer, messages, agent.config, tokenize=True, add_generation_prompt=True, **template)
    if prompt[:len(base)] != base:
        raise ValueError('Cannot isolate recovery generation prefix')
    window = agent.config.prompt_length + agent.config.response_length
    reserve = 0 if final else int(getattr(agent.config.plugin, 'final_answer_reserve', 0)) + int(getattr(agent.config.plugin, 'final_answer_safety_margin', 0))
    limit = min(512, max_new_tokens if max_new_tokens is not None else 512,
                agent.remaining_generation_tokens(), window - len(prompt) - reserve)
    if limit < 64:
        agent.budget_exhausted = True
        return None
    agent.task_recovery_requests = getattr(agent, 'task_recovery_requests', 0) + 1
    agent.task_format_retry_requests = getattr(agent, 'task_format_retry_requests', 0) + int(not final)
    record = dict(messages=messages, input_ids=list(prompt), max_tokens=limit,
                  max_len=len(prompt) + limit, phase='task_final' if final else 'task_format_retry',
                  control_enable_thinking=False, completion_kwargs={'structured_outputs': schema})
    agent.model_contexts.append(record)
    completion = await agent.llm_client.create_completion(prompt, uid=agent.context_uid,
        messages=messages, max_len=len(prompt) + limit, max_new_tokens=limit, structured_outputs=schema)
    if completion is None:
        return None
    message = completion['choices'][0]['message']
    text, output_ids = message['content'], list(message.get('raw_output_ids', []))
    record.update(response=text, output_ids=output_ids,
                  raw_output_text=message.get('raw_output_text'),
                  finish_reason=completion['choices'][0].get('finish_reason'))
    agent.total_request_tokens = getattr(agent, 'total_request_tokens', 0) + len(prompt) + len(output_ids)
    agent.peak_request_tokens = max(getattr(agent, 'peak_request_tokens', 0), len(prompt) + len(output_ids))
    # Keep original sampled tokens. The recorded generation prefix must match
    # this no-thinking request, not the ordinary cached thinking prefix.
    previous_prefix = agent.generation_prompt
    agent.generation_prompt = prompt[len(base):]
    try:
        agent.append({'role': 'assistant', 'content': text}, completion)
    finally:
        agent.generation_prompt = previous_prefix
    valid = bool(output_ids) and len(output_ids) <= limit and re.fullmatch(schema['regex'], text) is not None
    agent.task_recovery_pending = False
    agent.task_recovery_failed = not valid
    agent.task_recovery_successes = getattr(agent, 'task_recovery_successes', 0) + int(valid and not final)
    record['valid_native_tool'] = valid
    if not valid:
        print('[TASK FORMAT RECOVERY] constrained retry failed; stopping without execution')
        return None if final else '[Task action retry rejected] No valid native task call.'
    print('[TASK FORMAT RECOVERY] valid native tool; resuming task')
    return text.strip()
