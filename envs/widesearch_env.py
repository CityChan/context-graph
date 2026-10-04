"""Live-web tools for WideSearch. Gold answers and grading stay outside the agent."""
from __future__ import annotations

import collections
import json
import os
import time
from pathlib import Path

import httpx

from agents.agent_text import extract_fn_call


class WideSearchEnv:
    def __init__(self, config, tokenizer, ability):
        self.config, self.tokenizer, self.ability = config, tokenizer, ability
        self.stats = collections.Counter()
        self.instance_info = {}
        self.env_fail = self.is_finish = self.finish = False
        self.client = None
        self.pages = {}
        self.answer = ""

    async def init_env(self, item):
        import numpy as np
        value = np.asarray(item.non_tensor_batch["extra_info"], dtype=object)
        self.instance_info = dict(value.item() if value.ndim == 0 else value[0])
        if any(k in self.instance_info for k in ("evaluation", "answer", "gold", "gold_path")):
            raise ValueError("WideSearch agent input must not contain grading data")
        self.instance_info["problem_statement"] = self.instance_info["query"]
        key = os.environ.get("TAVILY_API_KEY")
        if not key:
            raise ValueError("WideSearch requires TAVILY_API_KEY for live search/extract")
        self.client = httpx.AsyncClient(base_url="https://api.tavily.com", timeout=60,
                                       headers={"Authorization": f"Bearer {key}"})

    def audit(self, record):
        path = self.instance_info.get("tool_log")
        if path:
            with Path(path).open("a", encoding="utf8") as stream:
                stream.write(json.dumps({"time": time.time(), **record}, ensure_ascii=False) + "\n")

    async def request(self, endpoint, payload):
        try:
            response = await self.client.post(endpoint, json=payload)
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict) or "results" not in data:
                raise ValueError("Malformed search provider response")
            self.audit({"endpoint": endpoint, "request": payload, "response": data})
            return data
        except Exception as exc:
            self.env_fail = True
            self.is_finish = self.finish = True
            self.stats["provider_errors"] += 1
            self.audit({"endpoint": endpoint, "error_type": type(exc).__name__})
            raise RuntimeError("WideSearch provider failure; not an agent score") from exc

    async def run_action(self, response):
        if self.is_finish:
            return {"action": "finish"}
        self.stats["action"] += 1
        call = extract_fn_call(response)
        if not call:
            return {"observation": "Use search, open_page, or finish with XML parameters."}
        name, args = call["function"], call["arguments"]
        if name == "finish":
            self.answer = args.get("answer", "").strip()
            self.is_finish = self.finish = True
            path = self.instance_info.get("prediction_path")
            if path:
                Path(path).write_text(self.answer, encoding="utf8")
            self.audit({"function": "finish", "answer": self.answer})
            return {"action": "finish", "observation": "Answer submitted for separate WideSearch grading."}
        if name == "search":
            query = args.get("query", "").strip()
            if not query:
                return {"observation": "search requires a nonempty query."}
            try:
                topk = max(1, min(20, int(args.get("topk", 10))))
            except ValueError:
                return {"observation": "topk must be an integer from 1 to 20."}
            data = await self.request("/search", {"query": query, "max_results": topk,
                                      "search_depth": "basic", "include_answer": False})
            rows = []
            for entry in data["results"]:
                url = entry["url"]
                docid = next((k for k, v in self.pages.items() if v == url), None)
                if docid is None:
                    docid = str(len(self.pages) + 1)
                    self.pages[docid] = url
                rows.append({"docid": docid, "url": url, "title": entry.get("title", ""),
                             "content": entry.get("content", "")[:2000]})
            self.stats["search"] += 1
            return {"observation": json.dumps(rows, ensure_ascii=False)}
        if name == "open_page":
            target = args.get("docid", args.get("url", "")).strip()
            url = self.pages.get(target, target)
            if not url.startswith(("https://", "http://")):
                return {"observation": "Use a docid from search or an HTTP(S) URL."}
            data = await self.request("/extract", {"urls": [url], "extract_depth": "basic"})
            self.stats["open_page"] += 1
            if not data["results"]:
                return {"observation": "Page extraction failed. Try another search result."}
            text = "\n".join(str(r.get("raw_content", "")) for r in data["results"])
            limit = int(getattr(self.config.plugin, "widesearch_page_chars", 20000))
            self.stats["output_truncations"] += int(len(text) > limit)
            return {"observation": url + "\n" + text[:limit]}
        return {"observation": f"Unknown web tool: {name}"}

    async def get_reward(self, item, messages, context):
        # The agent loop needs a reward slot; this is never exported as a benchmark score.
        return ("Official WideSearch grading pending", 0.0, {"ans_reward": 0.0})

    async def aclose(self):
        if self.client is not None:
            await self.client.aclose()
            self.client = None
