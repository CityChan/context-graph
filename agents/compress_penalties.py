"""Opt-in process penalties; heuristic cost signals, never task correctness."""
from collections import Counter
import re

from .agent_text import extract_fn_call


def similarity(left, right):
    a, b = set(re.findall(r'\w+', left.casefold())), set(re.findall(r'\w+', right.casefold()))
    return len(a & b) / max(1, len(a | b))


def apply_compress_penalties(agents, *, context_limit, branch_limit):
    stats = Counter()
    returns = []
    for agent in agents.values():
        searches, pages, branches, visible = [], set(), [], set()
        occupied = 0
        for index, (turn, completion) in enumerate(zip(agent.chat, agent.chat_completions)):
            content = str(turn.get('content', ''))
            ids = agent.tokenizer.encode(content, add_special_tokens=False)
            grams = [tuple(ids[i:i + 8]) for i in range(max(0, len(ids) - 7))]
            penalties = {}
            if completion is not None:
                choice = completion['choices'][0]
                # Missing EOS alone is insufficient: normal tool stop sequences
                # may omit it. Penalize only an explicitly reported output cap.
                raw = choice.get('message', {}).get('raw_output_ids', [])
                limit = choice.get('message', {}).get('extra_data', {}).get('max_tokens', 0)
                if choice.get('finish_reason') == 'length' or (
                    limit and len(raw) >= limit and raw[-1] != agent.tokenizer.eos_token_id
                ):
                    penalties['trunc'] = -1.0
                call = extract_fn_call(content)
                if call:
                    name, args = call['function'], call['arguments']
                    if name == 'search':
                        query = args.get('query', '')
                        if query and any(similarity(query, old) >= .8 for old in searches):
                            penalties['dup_search'] = -.3
                        searches.append(query)
                    elif name == 'open_page':
                        page = args.get('docid') or args.get('url')
                        if page and page in pages:
                            penalties['dup_search'] = -.3
                        if page:
                            pages.add(page)
                    elif name == 'branch':
                        task = args.get('prompt') or args.get('task') or args.get('description', '')
                        if len(branches) >= branch_limit:
                            penalties['dup_branch'] = -1.0
                        elif task and any(similarity(task, old) >= .5 for old in branches):
                            penalties['dup_branch'] = -.5
                        branches.append(task)
                    elif name == 'return':
                        message = args.get('message', '')
                        if len(agent.tokenizer.encode(message, add_special_tokens=False)) > 1500 or (
                            message and any(similarity(message, old) >= .8 for old in returns)
                        ):
                            penalties['ret_redund'] = -.3
                        returns.append(message)
                overlap = sum(g in visible for g in grams) / max(1, len(grams))
                if occupied > context_limit / 2 and len(ids) > 300 and overlap > .5:
                    penalties['redund'] = -.5 * overlap
                if penalties:
                    agent.set_process_reward(index, max(-1.0, sum(penalties.values())))
                    stats.update('compress_' + label for label in penalties)
            visible.update(grams)
            occupied += len(agent.chat_ids[index])
    return {'compress_' + label: stats['compress_' + label] for label in
            ('trunc', 'dup_search', 'dup_branch', 'redund', 'ret_redund')}
