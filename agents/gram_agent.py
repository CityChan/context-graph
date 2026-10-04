"""Independent GRAM episode executor with a separately configured memory model."""
from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
import json
import os
import time

from .gram_memory import DocumentStream, GraphMemory, cosine, entity_key, parse_action, token_f1, triples
from .gram_prompts import ACTOR, BCP_ACTOR, ENTITIES, RELATIONS, MAINTENANCE
from .gram_decoding import allowed_actions, action_regex


class MemoryBackendError(RuntimeError):
    pass


@dataclass(frozen=True)
class GramConfig:
    max_steps: int = 64
    max_episode_tokens: int = 32768
    max_step_tokens: int = 2048
    timeout_seconds: float = 1800
    search_hops: int = 2
    search_top_k: int = 12
    process_weight: float = 0.1
    entity_threshold: float = 0.9
    action_decoding: str = "unconstrained"

    def __post_init__(self):
        if self.action_decoding not in {"unconstrained", "xml_regex"}:
            raise ValueError("Unknown GRAM action_decoding")
        if min(self.max_steps, self.max_episode_tokens, self.max_step_tokens,
               self.timeout_seconds, self.search_hops, self.search_top_k) <= 0:
            raise ValueError("GRAM budgets must be positive")
        if self.process_weight < 0 or not 0 <= self.entity_threshold <= 1:
            raise ValueError("Invalid reward weight or entity threshold")


class MemoryBackend:
    """Frozen OpenAI-compatible helper; failures are infrastructure errors, never zero rewards."""
    def __init__(self, endpoint, model, *, embedding_endpoint=None, embedding_model=None,
                 timeout=120, max_tokens=2048, audit=None):
        import httpx
        if not endpoint or not model:
            raise ValueError("An explicit memory endpoint and model are required")
        if bool(embedding_endpoint) != bool(embedding_model):
            raise ValueError("Set both embedding endpoint and model")
        self.endpoint, self.model = endpoint.rstrip("/"), model
        self.embedding_endpoint, self.embedding_model = embedding_endpoint, embedding_model
        self.max_tokens, self.audit = max_tokens, audit or (lambda event: None)
        # Credentials are never stored in manifests or request audits.
        key = os.environ.get("GRAM_MEMORY_API_KEY", "")
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        self.client = httpx.AsyncClient(timeout=timeout, trust_env=False, headers=headers)
        self.embedding_cache = {}

    async def aclose(self):
        await self.client.aclose()

    async def json_call(self, operation, system, payload):
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
        started = time.monotonic()
        response = await self.client.post(self.endpoint + "/v1/chat/completions", json={
            "model": self.model, "messages": messages, "temperature": 0,
            "max_tokens": self.max_tokens, "chat_template_kwargs": {"enable_thinking": False}})
        response.raise_for_status()
        data = response.json()
        choice = data["choices"][0]
        text = choice["message"]["content"]
        self.audit({"kind": "memory_call", "operation": operation, "model": self.model,
                    "messages": messages, "response": text, "usage": data.get("usage"),
                    "finish_reason": choice.get("finish_reason"), "seconds": time.monotonic()-started})
        if choice.get("finish_reason") == "length":
            raise MemoryBackendError("Memory helper exhausted its completion budget")
        try:
            return json.loads(text)
        except (ValueError, TypeError) as exc:
            raise MemoryBackendError("Memory helper did not return strict JSON") from exc

    async def aliases(self, names, existing, threshold):
        canonical = {entity_key(name): name for name in existing}
        result = {}
        if self.embedding_model:
            missing = list(dict.fromkeys(n for n in [*existing, *names] if n not in self.embedding_cache))
            if missing:
                # Do not forward the memory server credential to a different embeddings host.
                import httpx
                async with httpx.AsyncClient(timeout=120, trust_env=False) as client:
                    response = await client.post(self.embedding_endpoint.rstrip("/") + "/v1/embeddings",
                                                 json={"model": self.embedding_model, "input": missing})
                    response.raise_for_status()
                    rows = sorted(response.json()["data"], key=lambda x: x["index"])
                if [r["index"] for r in rows] != list(range(len(missing))):
                    raise MemoryBackendError("Incomplete entity embeddings")
                for name, row in zip(missing, rows):
                    cosine(row["embedding"], row["embedding"])
                    self.embedding_cache[name] = row["embedding"]
                self.audit({"kind": "entity_embeddings", "model": self.embedding_model, "entities": missing})
        for name in names:
            key = entity_key(name)
            target = canonical.get(key)
            if target is None and self.embedding_model and canonical:
                scores = [(cosine(self.embedding_cache[name], self.embedding_cache[c]), c)
                          for c in canonical.values()]
                score, candidate = max(scores)
                if score > threshold:
                    target = candidate
            result[name] = target or name
            canonical.setdefault(key, result[name])
        return result

    async def edit(self, operation, content, document, question, graph, threshold):
        payload = {"question": question, "requested_facts": content, "document": document}
        try:
            if operation == "memory_insert":
                if content.strip().casefold() in {"none", "no relevant facts"}:
                    return {"add": [], "remove": [], "aliases": {}}
                entities = await self.json_call("entities", ENTITIES, {
                    **payload, "existing_entities": graph.entities})
                if not isinstance(entities, list) or any(not isinstance(e, str) or not e.strip() for e in entities):
                    raise ValueError("Entity extraction must return a list of nonempty strings")
                entities = list(dict.fromkeys(e.strip() for e in entities))
                add = triples(await self.json_call("relations", RELATIONS, {**payload, "entities": entities})) if entities else []
                if any(s not in entities or o not in entities for s, _, o in add):
                    raise ValueError("Relation extraction used an unverified entity")
                remove = []
            else:
                result = await self.json_call("maintenance", MAINTENANCE, {**payload, "graph": graph.snapshot()})
                if not isinstance(result, dict) or set(result) != {"add", "remove"}:
                    raise ValueError("Maintenance requires add and remove lists")
                add, remove = triples(result["add"]), triples(result["remove"])
            names = list(dict.fromkeys(x for e in add for x in (e[0], e[2])))
            aliases = await self.aliases(names, graph.entities, threshold)
            graph.apply(add, remove, document["id"], aliases)
            return {"add": add, "remove": remove, "aliases": aliases}
        except (ValueError, TypeError, KeyError) as exc:
            raise MemoryBackendError(str(exc)) from exc


async def run_episode(task, policy, memory, config: GramConfig, audit=None, *, retrieval=None):
    """policy(messages, token_limit) -> (text, exact sampled training segment).

    This function deliberately accepts no reference answers. Only the returned prediction
    can later be scored; neither actor nor helper can access the grading file.
    """
    if set(task) != {"task_id", "question", "documents"}:
        raise ValueError("Public GRAM tasks contain only task_id/question/documents")
    stream = DocumentStream(task["question"], list(task["documents"]), allow_empty=retrieval is not None)
    graph = GraphMemory()
    audit = audit or (lambda event: None)
    segments, valid, used, answer, search, feedback = [], [], 0, "", [], ""
    reason = "max_steps"
    external_searches = 0
    started = time.monotonic()
    async def execute():
        nonlocal used, answer, search, feedback, reason, external_searches
        for step in range(config.max_steps):
            budget = min(config.max_step_tokens, config.max_episode_tokens-used)
            if budget < 10:
                reason = "token_limit"
                break
            state = {"question": stream.question, "memory_obs": {
                "graph": graph.observation(), "search_paths": search, "feedback": feedback,
                "documents_consumed": stream.cursor, "documents_total": len(stream.documents)},
                "document_obs": stream.current}
            if retrieval is not None:
                state["budget"] = {"step": step, "max_steps": config.max_steps,
                                   "remaining_actor_tokens": config.max_episode_tokens-used}
            system = BCP_ACTOR if retrieval is not None else ACTOR
            if config.action_decoding == "xml_regex":
                state["action_decoding"] = config.action_decoding
                state["allowed_actions"] = allowed_actions(has_document=stream.current is not None,
                    external=retrieval is not None, external_searches=external_searches)
                system += "\nChoose exactly one of allowed_actions. Put any reasoning inside <think>...</think>; never before the tags.\n"
            messages = [{"role": "system", "content": system},
                        {"role": "user", "content": json.dumps(state, ensure_ascii=False)}]
            text, segment = await policy(messages, budget)
            if not segment.get("response_ids"):
                raise RuntimeError("Actor returned no sampled token IDs")
            used += sum(segment["response_mask"])
            segments.append(segment)
            event = {"kind": "action", "step": step, "document_index": stream.cursor,
                     "messages": messages, "response": text, "sampled_tokens": sum(segment["response_mask"])}
            try:
                operation, content = parse_action(text, external_search=retrieval is not None)
            except ValueError as exc:
                valid.append(0)
                feedback = str(exc)
                audit({**event, "valid_format": False, "error": feedback})
                continue
            valid.append(1)
            feedback = ""
            if operation == "answer" and retrieval is not None and not external_searches:
                feedback = "Perform an external corpus search before answering."
            elif operation == "answer":
                answer, reason = content, "answer"
            elif operation in {"search", "open_page"}:
                if stream.current is not None:
                    feedback = "Consume the current observation with Insert or Update first."
                else:
                    observation = await retrieval(operation, content)
                    document = {"id": f"retrieval-{len(stream.documents)}", "title": f"{operation}: {content}", "text": observation}
                    stream.documents.append(document)
                    external_searches += int(operation == "search")
                    event["retrieved_document"] = document
            elif operation == "memory_search":
                search = graph.search(content, config.search_hops, config.search_top_k)
            elif stream.current is None:
                feedback = "No document remains. Search memory or answer."
            else:
                edit = await memory.edit(operation, content, stream.current, stream.question, graph,
                                         config.entity_threshold)
                stream.advance()
                search = []  # Invalidate stale paths after every graph mutation.
                event["edit"] = edit
            audit({**event, "operation": operation, "valid_format": True, "graph": graph.snapshot()})
            if reason == "answer":
                break
    try:
        async with asyncio.timeout(config.timeout_seconds):
            await execute()
    except TimeoutError:
        # A stalled API/backend is not an ordinary low-quality policy answer.
        raise MemoryBackendError("GRAM episode exceeded its wall-clock deadline")
    return {"task_id": task["task_id"], "prediction": answer, "termination_reason": reason,
            "format_reward": sum(valid)/len(valid) if valid else 0.0,
            "steps": len(segments), "policy_tokens": used, "documents_consumed": stream.cursor,
            "external_searches": external_searches,
            "graph": graph.snapshot(), "elapsed_seconds": time.monotonic()-started,
            "config": asdict(config), "segments": segments}


def score_episode(result, answers, process_weight=0.1):
    if not isinstance(answers, list) or not answers or any(not isinstance(a, str) or not a.strip() for a in answers):
        raise ValueError("Nonempty reference answers are required")
    outcome = token_f1(result["prediction"], answers) if result["termination_reason"] == "answer" else 0.0
    return {"answer_f1": outcome, "format_reward": result["format_reward"],
            "training_reward": outcome + process_weight*result["format_reward"]}


def make_policy(client, tokenizer, rollout_config):
    """Export exact sampled prefixes, without silently truncating a document or graph."""
    from .utils import Agent, _apply_chat_template
    async def policy(messages, budget):
        raw = _apply_chat_template(tokenizer, messages, rollout_config,
                                   add_generation_prompt=True, tokenize=True, enable_thinking=False)
        if len(raw) > rollout_config.prompt_length:
            raise MemoryBackendError("GRAM prompt exceeds prompt_length; increase it or explicitly change the protocol")
        agent = Agent(client, messages, tokenizer, rollout_config, prompt_turn=len(messages),
                      chat_template_kwargs={"enable_thinking": False})
        budget = min(budget, rollout_config.response_length-len(agent.get_generation_prompt()))
        if budget < 10:
            raise ValueError("response_length cannot hold an action")
        state = json.loads(messages[-1]["content"])
        completion_kwargs = {}
        if state.get("action_decoding") == "xml_regex":
            completion_kwargs["structured_outputs"] = {"regex": action_regex(state["allowed_actions"])}
        text = await agent.step(max_new_tokens=budget, completion_kwargs=completion_kwargs)
        if text is None:
            raise RuntimeError("Actor completion missing")
        return text, await agent.get_data()
    return policy
