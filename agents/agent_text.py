"""Shared legacy tool-call parsing and branch-response text helpers."""

import re


def extract_fn_call(text):
    if text is None:
        return None
    func_matches = re.findall(r'<function=([^>]+)>', text)
    if not func_matches:
        return None
    last_function = func_matches[-1]
    last_func_pos = text.rfind(f'<function={last_function}>')
    text_after_last_func = text[last_func_pos:]
    params = dict(re.findall(r'<parameter=([^>]+)>(.*?)</parameter>', text_after_last_func, re.DOTALL))
    return {'function': last_function, 'arguments': params}


def extract_summary(text: str) -> str:
    matches = re.findall(r'<summary>(.*?)</summary>', text, re.DOTALL)
    return matches[-1].strip() if matches else None


def clean_response(response):
    # vLLM returns None when token budget for a turn is negative (rollout skipped);
    # downstream code uses the cleaned text as a fallback branch summary, so emit
    # a clear placeholder rather than raising TypeError on None.
    if not response:
        return '[no response — turn skipped (token budget exhausted)]'
    if '<function=return>' in response:
        response = response.split('<function=return>')[-1]
    else:
        response = re.split(r'<\[[^\]]+\]>', response)[-1]
    return response
