"""Environment contracts for the evaluation-only AgentFold/SUPO adapters."""
import json


def task_kind(ability):
    if ability in {'LocalSearch', 'GAIA'}:
        return 'search'
    if ability.startswith('SWEVerified@'):
        return 'swe'
    if ability.startswith('DiscoveryWorld@'):
        return 'discoveryworld'
    if ability.startswith('ScienceWorld@'):
        return 'scienceworld'
    raise ValueError(f'Unsupported baseline ability: {ability}')


def task_chat(kind, task, item, method, create_chat, *, prompt_profile='legacy'):
    if kind == 'search':
        return create_chat(task, 'search_agentfold' if method == 'agentfold' else 'search_single', item)
    if kind == 'swe':
        from .prompts_swe import create_chat_swe
        messages = create_chat_swe(task, 'code')
    elif kind == 'scienceworld':
        messages = create_chat(task, 'scienceworld', item, scienceworld_prompt_profile=prompt_profile)
        if method == 'agentfold':
            messages[0]['content'] = messages[0]['content'].replace(
                'Output exactly one XML tool call per turn, with no prose or markdown.',
                'Emit exactly one environment action per response; obey the memory protocol below.').replace(
                'Output exactly one function call, with no prose or markdown before or after it.',
                'Emit exactly one environment action after any required memory control block.')
    else:
        messages = create_chat(task, 'discoveryworld', item)
        # Memory adaptations emit their own control text, never branch/graph ops.
        messages[0]['content'] = messages[0]['content'].replace(
            'Output only one XML function call, with no prose or markdown.',
            'Emit exactly one environment action per response; obey the memory protocol below.')
    messages[0]['content'] += '\nOnly one environment tool per response. The latest observation supersedes historical state.'
    return messages


def adapt_fold_text(text, kind):
    if kind == 'search':
        return text
    text = text.replace('search/open_page', 'python_exec' if kind == 'swe' else 'action')
    if kind in ('discoveryworld', 'scienceworld'):
        text = text.replace('A finish call needs no compression.',
                            'Do not call finish; the simulator determines termination.')
    return text


def tool_schema(kind):
    if kind == 'swe':
        return {'python_exec': {'code'}, 'finish': {'message'}}, {'python_exec': ('code',), 'finish': ('message',)}
    if kind in ('discoveryworld', 'scienceworld'):
        return {'action': {'command'}}, {'action': ('command',)}
    return ({'search': {'query', 'topk'}, 'open_page': {'docid', 'url'},
             'finish': {'answer', 'explanation', 'confidence'}},
            {'search': ('query',), 'open_page': ('docid', 'url'), 'finish': ('answer',)})


def validate_call(kind, call, name, args):
    if kind == 'search':
        from envs.local_search import extract_fn_call
        if extract_fn_call(call) != [{'function': name, 'arguments': args}]:
            raise ValueError('ambiguous tool markup inside arguments')
    elif kind == 'swe':
        from .agent_text import extract_fn_call
        parsed = extract_fn_call(call)
        if parsed != {'function': name, 'arguments': args}:
            raise ValueError('ambiguous SWE tool markup')
    elif kind == 'scienceworld':
        from envs.scienceworld_env import ScienceWorldEnv
        command = args['command'].strip()
        if '\n' in command or '\r' in command:
            raise ValueError('ScienceWorld requires one single-line command')
        if ScienceWorldEnv._parse_fn_call(call) != {'function': name, 'arguments': {'command': command}}:
            raise ValueError('Ambiguous ScienceWorld command markup')
    elif not isinstance(json.loads(args['command']), dict):
        raise ValueError('DiscoveryWorld command must be a JSON object')


def provenance(kind):
    return {'task_protocol': kind + '_v1', 'stateful': kind != 'search',
            'environment_finish': kind in ('discoveryworld', 'scienceworld')}
