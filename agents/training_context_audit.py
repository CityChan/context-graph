"""Fail closed when exported RL tokens no longer match the sampled prefixes."""
import math

from .exact_context import checksum


def record_sample(agent, prompt, completion):
    message = completion['choices'][0]['message']
    output = list(message['raw_output_ids'])
    logprobs = list(message['response_log_probs'])
    if not output or len(output) != len(logprobs) or not all(math.isfinite(x) for x in logprobs):
        raise ValueError('RL sample requires aligned finite output logprobs')
    backend_prompt = message.get('extra_data', {}).get('input_ids')
    if backend_prompt is not None and list(backend_prompt) != list(prompt):
        raise ValueError('RL backend prompt differs from requested context')
    audits = getattr(agent, 'sampled_context_audits', {})
    audits[len(agent.chat)] = dict(prefix=checksum(list(prompt)), output=checksum(output),
                                   logprobs=checksum(logprobs), output_length=len(output))
    agent.sampled_context_audits = audits


def validate_segment(agent, start, end, data):
    width = len(data['response_ids'])
    aligned = ('response_mask', 'response_logprobs', 'process_reward_mask',
               'graph_edit_credit_mask', 'graph_decision_mask', 'graph_invalid_penalty_mask')
    if any(len(data[key]) != width for key in aligned):
        raise ValueError('RL exported token fields have inconsistent lengths')
    mapping = data['response_turn_token_indices']
    expected_turns = {turn for turn in range(start, end) if any(agent.token_mask[turn])}
    if set(mapping) != expected_turns:
        raise ValueError('RL export dropped or invented sampled turns')
    supervised = set()
    for turn, indices in mapping.items():
        saved = getattr(agent, 'sampled_context_audits', {}).get(turn)
        if saved is None or len(indices) != saved['output_length']:
            raise ValueError('RL export truncated sampled output or lacks provenance')
        if indices != list(range(indices[0], indices[0] + len(indices))):
            raise ValueError('RL sampled span is not contiguous')
        if checksum(data['prompt_ids'] + data['response_ids'][:indices[0]]) != saved['prefix']:
            raise ValueError('RL sampled conditioning prefix changed')
        if checksum([data['response_ids'][i] for i in indices]) != saved['output']:
            raise ValueError('RL sampled output changed')
        if checksum([data['response_logprobs'][i] for i in indices]) != saved['logprobs']:
            raise ValueError('RL sampled rollout logprobs changed')
        info = agent.additional_info[turn] or {}
        labels = {'process_reward_mask': info.get('process_reward', 0),
                  'graph_edit_credit_mask': info.get('graph_edit_credit', 0),
                  'graph_decision_mask': int('graph_edit_credit' in info),
                  'graph_invalid_penalty_mask': info.get('graph_invalid_penalty', 0)}
        for field, expected in labels.items():
            if any(data[field][i] != expected for i in indices):
                raise ValueError('RL credit/penalty moved to a different sampled turn')
        if supervised.intersection(indices):
            raise ValueError('RL sampled spans overlap')
        supervised.update(indices)
    if supervised != {i for i, mask in enumerate(data['response_mask']) if mask == 1}:
        raise ValueError('RL response mask differs from sampled spans')
    for i, mask in enumerate(data['response_mask']):
        if mask not in (0, 1) or (not mask and any(data[k][i] != 0 for k in aligned[2:])):
            raise ValueError('RL masks label non-policy tokens')
    return len(mapping)


def pack_verified_segment(agent, data):
    """Keep normal segments; split oversized ones at native request boundaries.

    A retained prefix beyond the fixed prompt allocation occupies masked response
    slots. No sampled token or conditioning token is discarded.
    """
    prompt_limit, response_limit = agent.config.prompt_length, agent.config.response_length
    if len(data['prompt_ids']) <= prompt_limit and len(data['response_ids']) <= response_limit:
        yield data
        return
    fields = ('response_logprobs', 'response_mask', 'process_reward_mask',
              'graph_edit_credit_mask', 'graph_decision_mask', 'graph_invalid_penalty_mask')
    for turn, indices in data['response_turn_token_indices'].items():
        prefix = data['prompt_ids'] + data['response_ids'][:indices[0]]
        retained = prefix[prompt_limit:]
        if len(retained) + len(indices) > response_limit:
            raise ValueError('RL native request exceeds fixed training capacity')
        row = dict(data)
        row['prompt_ids'] = prefix[:prompt_limit]
        row['response_ids'] = retained + [data['response_ids'][i] for i in indices]
        for field in fields:
            row[field] = [0] * len(retained) + [data[field][i] for i in indices]
        row['response_turn_token_indices'] = {turn: list(range(len(retained), len(row['response_ids'])))}
        row['verified_sampled_turns'] = validate_segment(agent, turn, turn + 1, row)
        yield row
