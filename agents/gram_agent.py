"""Independent GRAM episode executor with a separately configured memory model."""
from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
import json
import os
import re
import time

from .gram_memory import DocumentStream, GraphMemory, cosine, entity_key, parse_action, token_f1, triples, memory_output_schema, validate_memory_output
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
    bcp_progress_limit: int = 0

    def __post_init__(self):
        if self.action_decoding not in {"unconstrained", "xml_regex"}:
            raise ValueError("Unknown GRAM action_decoding")
        if type(self.bcp_progress_limit) is not int or self.bcp_progress_limit < 0:
            raise ValueError("bcp_progress_limit must be a nonnegative integer")
        if self.bcp_progress_limit and self.action_decoding != "xml_regex":
            raise ValueError("BC-P progress guard requires xml_regex decoding")
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
        structured_outputs = {"json": memory_output_schema(operation,
            entities=payload["entities"] if operation == "relations" else None)}
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
        started = time.monotonic()
        response = await self.client.post(self.endpoint + "/v1/chat/completions", json={
            "model": self.model, "messages": messages, "temperature": 0,
            "max_tokens": self.max_tokens, "chat_template_kwargs": {"enable_thinking": False},
            "structured_outputs": structured_outputs})
        response.raise_for_status()
        data = response.json()
        choice = data["choices"][0]
        text = choice["message"]["content"]
        self.audit({"kind": "memory_call", "operation": operation, "model": self.model,
                    "messages": messages, "response": text, "usage": data.get("usage"),
                    "structured_outputs": structured_outputs,
                    "finish_reason": choice.get("finish_reason"), "seconds": time.monotonic()-started})
        if choice.get("finish_reason") == "length":
            raise MemoryBackendError(f"Memory helper {operation} exhausted its completion budget")
        try:
            value = json.loads(text)
        except (ValueError, TypeError) as exc:
            raise MemoryBackendError(f"Memory helper {operation} did not return strict JSON") from exc
        try:
            validate_memory_output(operation, value)
        except ValueError as exc:
            raise MemoryBackendError(f"Memory helper {operation}: {exc}") from exc
        return value

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
        # The actor selects useful facts. Supplying its research question here
        # made the frozen extractor discard intermediate evidence when it could
        # not answer that question (saved-request ablation: qhLVNF).
        # Keep the public edit signature; never forward question to any stage.
        payload = {"requested_facts": content, "document": document}
        try:
            if operation == "memory_insert":
                if content.strip().casefold() in {"none", "no relevant facts"}:
                    return {"add": [], "remove": [], "aliases": {}, "extraction_status": "explicit_skip"}
                entities = await self.json_call("entities", ENTITIES, {
                    **payload, "existing_entities": graph.entities})
                if not isinstance(entities, list) or any(not isinstance(e, str) or not e.strip() for e in entities):
                    raise ValueError("Entity extraction must return a list of nonempty strings")
                entities = list(dict.fromkeys(e.strip() for e in entities))
                add = triples(await self.json_call("relations", RELATIONS, {**payload, "entities": entities})) if entities else []
                unverified = sorted({name for s, _, o in add for name in (s, o) if name not in entities})
                if unverified:
                    raise ValueError(f"Relation extraction used an unverified entity: {unverified!r}")
                remove = []
                extraction_status = "triples_extracted" if add else "no_relations" if entities else "no_entities"
            else:
                result = await self.json_call("maintenance", MAINTENANCE, {**payload, "graph": graph.snapshot()})
                if not isinstance(result, dict) or set(result) != {"add", "remove"}:
                    raise ValueError("Maintenance requires add and remove lists")
                add, remove = triples(result["add"]), triples(result["remove"])
                extraction_status = "maintenance_changes" if add or remove else "no_maintenance_changes"
            names = list(dict.fromkeys(x for e in add for x in (e[0], e[2])))
            aliases = await self.aliases(names, graph.entities, threshold)
            graph.apply(add, remove, document["id"], aliases)
            return {"add": add, "remove": remove, "aliases": aliases, "extraction_status": extraction_status}
        except (ValueError, TypeError, KeyError) as exc:
            raise MemoryBackendError(f"Memory helper {operation}: {exc}") from exc


async def run_episode(task, policy, memory, config: GramConfig, audit=None, *, retrieval=None):
    """policy(messages, token_limit) -> (text, exact sampled training segment).

    This function deliberately accepts no reference answers. Only the returned prediction
    can later be scored; neither actor nor helper can access the grading file.
    """
    if set(task) != {"task_id", "question", "documents"}:
        raise ValueError("Public GRAM tasks contain only task_id/question/documents")
    if config.bcp_progress_limit and retrieval is None:
        raise ValueError("BC-P progress guard requires external retrieval")
    stream = DocumentStream(task["question"], list(task["documents"]), allow_empty=retrieval is not None)
    graph = GraphMemory()
    audit = audit or (lambda event: None)
    segments, valid, used, answer, search, feedback = [], [], 0, "", [], ""
    reason = "max_steps"
    external_searches = 0
    memory_searches = guard_blocks = 0
    last_memory_search = None
    retrieval_history, seen_retrievals = [], set()
    last_memory_edit = None
    duplicate_retrievals = no_change_edits = 0
    started = time.monotonic()
    async def execute():
        nonlocal used, answer, search, feedback, reason, external_searches
        nonlocal memory_searches, guard_blocks, last_memory_search
        nonlocal last_memory_edit, duplicate_retrievals, no_change_edits
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
                    external=retrieval is not None, external_searches=external_searches,
                    progress_limit=config.bcp_progress_limit, has_memory=bool(graph.edges),
                    memory_searches=memory_searches)
                system += "\nChoose exactly one of allowed_actions. Put any reasoning inside <think>...</think>; never before the tags.\n"
            if config.bcp_progress_limit:
                state["memory_obs"].update(last_memory_search=last_memory_search,
                    memory_searches_since_progress=memory_searches,
                    memory_search_limit=config.bcp_progress_limit,
                    last_memory_edit=last_memory_edit)
                # Operational history, not an alternative store of answer evidence.
                state["retrieval_history"] = retrieval_history[-8:]
                system += ("\nInternal memory_search only reads stored facts and cannot discover new evidence. "
                           "It is unavailable for an empty graph or after the memory search limit. "
                           "Consume a pending document before answering or retrieving more. "
                           "When stored facts do not suffice and no document is pending, use external search/open_page. "
                           "retrieval_history records previous requests and returned docids, not answer evidence. "
                           "Do not repeat a previous external request: refine the query or open a returned docid. "
                           "Check last_memory_edit: consuming a document does not mean its facts were saved. "
                           "If no edges were added, open the source page or change the query to obtain supported facts. "
                           "Keep actions concise; insert only supported facts, not speculation.\n")
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
            if config.bcp_progress_limit and operation not in state["allowed_actions"]:
                guard_blocks += 1
                feedback = "Action unavailable under BC-P progress guard; choose from allowed_actions."
                audit({**event, "operation": operation, "valid_format": True,
                       "executed": False, "error": feedback, "allowed_actions": state["allowed_actions"]})
                continue
            if operation == "answer" and retrieval is not None and not external_searches:
                feedback = "Perform an external corpus search before answering."
            elif operation == "answer":
                answer, reason = content, "answer"
            elif operation in {"search", "open_page"}:
                if stream.current is not None:
                    feedback = "Consume the current observation with Insert or Update first."
                else:
                    # Normalize only query case/whitespace; docids are case-sensitive.
                    key = (operation, " ".join(content.split()).casefold() if operation == "search" else content.strip())
                    if config.bcp_progress_limit and key in seen_retrievals:
                        duplicate_retrievals += 1
                        guard_blocks += 1
                        feedback = "External request already executed; no new retrieval performed. Change the query or open a different returned docid."
                        audit({**event, "operation": operation, "valid_format": True,
                               "executed": False, "error": feedback})
                        continue
                    observation = await retrieval(operation, content)
                    seen_retrievals.add(key)
                    document = {"id": f"retrieval-{len(stream.documents)}", "title": f"{operation}: {content}", "text": observation}
                    stream.documents.append(document)
                    external_searches += int(operation == "search")
                    memory_searches = 0
                    last_memory_search = None
                    event["retrieved_document"] = document
                    if config.bcp_progress_limit:
                        retrieval_history.append({"step": step, "operation": operation,
                            "request": content[:512], "document_id": document["id"],
                            "returned_docids": list(dict.fromkeys(re.findall(r"(?m)^docid:\s*([^\s]+)", observation)))[:5]})
                        del retrieval_history[:-8]
            elif operation == "memory_search":
                search = graph.search(content, config.search_hops, config.search_top_k)
                memory_searches += 1
                last_memory_search = {"query": content, "path_count": len(search)}
                if config.bcp_progress_limit:
                    feedback = f"Internal memory returned {len(search)} paths; it does not fetch new evidence."
            elif stream.current is None:
                feedback = "No document remains. Search memory or answer."
            else:
                before_edges = set(graph.edges)
                edit = await memory.edit(operation, content, stream.current, stream.question, graph,
                                         config.entity_threshold)
                added, removed = len(set(graph.edges) - before_edges), len(before_edges - set(graph.edges))
                no_change_edits += int(not added and not removed)
                last_memory_edit = {"document_id": stream.current["id"],
                    "extraction_status": edit.get("extraction_status", "unreported"),
                    "new_edges": added, "removed_edges": removed, "graph_edges": len(graph.edges)}
                event["memory_effect"] = last_memory_edit
                if config.bcp_progress_limit:
                    feedback = f"Memory edit saved {added} new edges and removed {removed}; graph has {len(graph.edges)} edges."
                    if not added and not removed:
                        feedback += " No new facts were saved. Do not repeat the same retrieval; seek different evidence or open its source page."
                stream.advance()
                memory_searches = 0
                last_memory_search = None
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
            "progress_guard_blocks": guard_blocks,
            "duplicate_retrievals_blocked": duplicate_retrievals,
            "no_change_memory_edits": no_change_edits,
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
