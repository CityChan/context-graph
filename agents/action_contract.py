"""Constrained policy actions; raw JSON stays in the learned trajectory."""
import json

from .baseline_task_protocol import task_kind

PROFILE = 'typed_action_json_v1'
# A launch floor, not a guarantee that arbitrary argument text will fit.
MIN_ACTION_TOKENS = 256
BRANCH_RETURN_RESERVE = 768
BRANCH_HANDOFF_RESERVE = 1536
BRANCH_RETURN_PROMPT = ('Branch budget is ending. Use return now with a concise report of observed '
                        'findings, source IDs and unresolved work. Do not claim the task is solved '
                        'without evidence. Choose JSON tool=return, arguments.message (at most 256 characters).')
FINAL_PROMPT = ('FINAL ANSWER REQUIRED NOW. Use only evidence already visible to submit your '
                'best-supported answer; state uncertainty when evidence is incomplete. '
                'Choose JSON tool=finish with arguments.answer, and optionally a brief '
                'arguments.explanation with source IDs and arguments.confidence. Do not research further.')


def validate_training_contract(config):
    # PPO recomputes the old-policy likelihood on the original JSON sequence.
    # Do not mix backend-dependent guided log-probs into that denominator.
    actor = config.actor_rollout_ref.get('actor', {})
    correction = config.algorithm.get('rollout_correction') or {}
    if (actor.get('use_rollout_log_probs', False) or correction.get('bypass_mode', False)
            or any(correction.get(key) is not None for key in ('rollout_is', 'rollout_rs', 'rollout_token_veto_threshold'))):
        raise ValueError('typed actions require recomputed old-policy log-probs; rollout bypass is unsupported')


PROMPT = '''
Action interface (overrides XML examples above): choose exactly one tool and its
arguments as a JSON object {"tool":"name","arguments":{...}}. The framework
constrains syntax and serializes the environment call. Do not emit XML, markdown
or a thinking block. Choose actions for their usefulness, not to satisfy a quota.
merge uses a node_ids array of 2-6 IDs. graph_read offset and search topk are
integers. DiscoveryWorld action.command is an object, not a JSON-encoded string.
Main may branch; a branch may return but cannot branch or finish. For simulators,
the environment ends the episode; finish is unavailable. Graph edits are your
decisions; invalid targets are rejected without silently changing your choice.
'''


THINKING_PROMPT = PROMPT.replace(
    'constrains syntax and serializes the environment call. Do not emit XML, markdown\nor a thinking block.',
    'validates the final JSON and serializes the environment call. Reason in the thinking\n'
    'channel if needed, then end thinking and emit only the JSON action, without XML or markdown.')


def obj(properties, required=None):
    return {'type': 'object', 'properties': properties,
            'required': list(properties if required is None else required), 'additionalProperties': False}


def string(max_length=4096):
    return {'type': 'string', 'minLength': 1, 'maxLength': max_length}


def action(name, properties, required=None):
    return obj({'tool': {'const': name}, 'arguments': obj(properties, required)})


def _validate(value, schema):
    """Validate exactly the schema subset emitted below, including strict integers."""
    if 'anyOf' in schema:
        for choice in schema['anyOf']:
            try:
                _validate(value, choice)
                return
            except ValueError:
                pass
        raise ValueError('action or arguments do not match the available tools')
    if 'const' in schema and value != schema['const']:
        raise ValueError('unexpected constant')
    if 'enum' in schema and value not in schema['enum']:
        raise ValueError('unknown value')
    kind = schema.get('type')
    if kind == 'object':
        if not isinstance(value, dict) or set(value) - schema['properties'].keys() or not set(schema['required']) <= value.keys():
            raise ValueError('missing or unknown arguments')
        for key, child in value.items():
            _validate(child, schema['properties'][key])
    elif kind == 'string':
        if not isinstance(value, str) or not schema.get('minLength', 0) <= len(value) <= schema.get('maxLength', len(value)):
            raise ValueError('expected nonempty string')
    elif kind == 'integer':
        if type(value) is not int or value < schema.get('minimum', value):
            raise ValueError('expected integer in range')
    elif kind == 'array':
        if not isinstance(value, list) or not schema['minItems'] <= len(value) <= schema['maxItems']:
            raise ValueError('invalid array size')
        for child in value:
            _validate(child, schema['items'])


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate key')
        result[key] = value
    return result


class ActionContract:
    def __init__(self, ability, env, *, branch=False):
        self.kind = task_kind(ability)
        self.env, self.branch = env, branch

    def schema(self, *, final=False):
        text = string()
        finish = action('finish', {'answer': text, 'explanation': text, 'confidence': text}, ['answer']) if self.kind == 'search' else action('finish', {'message': text})
        if final:
            if self.branch:
                return action('return', {'message': string(256)})
            if self.kind in ('scienceworld', 'discoveryworld'):
                raise ValueError('finish is unavailable for this agent')
            return (action('finish', {'answer': string(256), 'explanation': string(512),
                                     'confidence': string(16)}, ['answer'])
                    if self.kind == 'search' else action('finish', {'message': string(512)}))
        choices = [action('merge', {'node_ids': {'type': 'array', 'items': text, 'minItems': 2, 'maxItems': 6}, 'summary': text}),
                   action('add_edge', {'source': text, 'target': text, 'relation': {'enum': ['causal', 'semantic', 'temporal']}}),
                   action('graph_read', {'node_id': text, 'offset': {'type': 'integer', 'minimum': 0}}, []),
                   action('graph_compact', {})]
        choices += [action(name, {'node_id': text}) for name in ('prune', 'select', 'recall')]
        choices.append(action('return', {'message': text}) if self.branch else action('branch', {'description': text, 'prompt': text}))
        if self.kind == 'search':
            choices += [action('search', {'query': text, 'topk': {'type': 'integer', 'minimum': 1}}, ['query']),
                        action('open_page', {'docid': text}), action('open_page', {'url': text})]
        elif self.kind == 'swe':
            choices.append(action('python_exec', {'code': text}))
        elif self.kind == 'scienceworld':
            choices.append(action('action', {'command': text}))
        else:
            if self.env._env.isAgentInDialog(0):
                commands = [obj({'chosen_dialog_option_int': {'type': 'integer'}})]
            else:
                commands = []
                for name, spec in self.env._actions.items():
                    arg = text if name in ('MOVE_DIRECTION', 'ROTATE_DIRECTION', 'TELEPORT_TO_LOCATION') else {'type': 'integer'}
                    commands.append(obj({'action': {'const': name}, **{key: arg for key in spec['args']}}))
            choices.append(action('action', {'command': {'anyOf': commands}}))
        if not self.branch and self.kind in ('search', 'swe'):
            choices.append(finish)
        return {'anyOf': choices}

    def decode(self, response, schema):
        payload = json.loads(response, object_pairs_hook=_unique_object,
                             parse_constant=lambda x: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
        _validate(payload, schema)
        name, args = payload['tool'], dict(payload['arguments'])
        if name == 'merge':
            if any(',' in node for node in args['node_ids']):
                raise ValueError('each node_ids element must identify one node')
            args['node_ids'] = ','.join(args['node_ids'])
        if name == 'action' and self.kind == 'discoveryworld':
            args['command'] = json.dumps(args['command'], ensure_ascii=False)
        # The legacy adapters use delimiter parsing rather than an XML parser.
        # Reject ambiguous payloads; never execute injected or partial calls.
        values = {key: str(value) for key, value in args.items()}
        forbidden = ('<function', '</function', '<parameter', '</parameter', '<think', '</think', '<tool_call')
        if any(marker in value for value in values.values() for marker in forbidden):
            raise ValueError('argument contains reserved transport delimiters')
        if self.kind == 'scienceworld' and name == 'action' and any(c in values['command'] for c in '\r\n'):
            raise ValueError('ScienceWorld command must be a single line')
        wire = f'<function={name}>' + ''.join(f'<parameter={key}>{value}</parameter>' for key, value in values.items()) + '</function>'
        return payload, wire
