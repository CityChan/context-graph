"""External BC-P retrieval for GRAM; graph search remains inside gram_agent."""
import asyncio
import json
from types import SimpleNamespace

from envs.local_search import LocalSearch, judge


RETRIEVAL_LIMITS = dict(search_topk_cap=5, search_snippet_words=128,
                        search_snippet_chars=2000, open_page_words=4096, open_page_chars=48000)


class BcpRetrieval:
    """Reuse the existing environment's ranking, repeat snippets and truncation."""
    def __init__(self):
        config = SimpleNamespace(plugin=SimpleNamespace(**RETRIEVAL_LIMITS))
        self.env = LocalSearch(config, None, "bcp")
        # run_action search/open needs these fields, but never needs private labels.
        self.env.search_skill_context = None
        self.env.search_skill_context_injected = False

    async def __call__(self, operation, content):
        if operation not in {"search", "open_page"}:
            raise ValueError("Only external search/open_page may reach the corpus")
        arguments = {"query": content, "topk": 5} if operation == "search" else {"docid": content}
        # JSON serialization prevents tool syntax embedded in a query from injecting actions.
        payload = json.dumps({"name": operation, "arguments": arguments}).replace("<", "\\u003c").replace(">", "\\u003e")
        response = await self.env.run_action('<tool_call>' + payload + '</tool_call>')
        return response["observation"]

    async def aclose(self):
        await self.env.client.close()


async def score_bcp(question, answer, prediction):
    audit = []
    for attempt, delay in enumerate((0, 15, 60), 1):
        if delay:
            await asyncio.sleep(delay)
        before = len(audit)
        score = await judge(question, answer, prediction, audit_sink=audit)
        for row in audit[before:]:
            row["outer_attempt"] = attempt
        method = audit[-1].get("judge_method") if len(audit) > before else None
        if method in {"strict_em", "blank_prediction", "llm_judge"}:
            return {"score": score, "judge_audit": audit}
        # Keep raw grader replies in failure logs; never retry a valid negative.
        print("GRAM_JUDGE_FAILURE " + json.dumps(audit[before:], ensure_ascii=False), flush=True)
        if method != "llm_parse_failure":
            break
    raise RuntimeError("BC-P judge failed; this is not a policy failure")


def load_tasks(path, samples, seed, shard_index=0, shard_count=1):
    import pandas as pd
    from scripts.eval_bcp_qwen38 import select_indices
    rows = pd.read_parquet(path).to_dict("records")
    indices = select_indices(len(rows), samples, seed)[shard_index::shard_count]
    tasks, refs = [], {}
    for index in indices:
        extra = rows[index]["extra_info"]
        question, answer = extra["query"], extra["answer"]
        if not all(isinstance(v, str) and v.strip() for v in (question, answer)):
            raise ValueError(f"Invalid BC-P question/reference at row {index}")
        identity = f"bcp-{index}"
        tasks.append({"task_id": identity, "question": question, "documents": []})
        refs[identity] = answer
    if not tasks:
        raise ValueError("Empty BC-P shard")
    return tasks, refs, indices
