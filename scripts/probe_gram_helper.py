"""Replay saved entity requests with isolated ablations; no actor, retriever or judge."""
import argparse
import asyncio
import copy
import json
from pathlib import Path

from agents.gram_agent import MemoryBackend
from agents.gram_memory import GraphMemory, memory_output_schema, validate_memory_output
from agents.gram_prompts import ENTITIES, RELATIONS
from scripts.gram_helper_checks import check_helper_controls

FOOTER = '\n\n* Please reflect on the information we have obtained, and keep searching for additional information if we still can not answer the question. Do not give the answer if the information is still not enough.'


def variants(event):
    original = event["messages"]
    yield "original", copy.deepcopy(original), False
    for name in ("without_existing_entities", "without_question", "without_tool_footer"):
        messages = copy.deepcopy(original)
        payload = json.loads(messages[-1]["content"])
        if name == "without_tool_footer":
            payload["document"]["text"] = payload["document"]["text"].removesuffix(FOOTER)
        else:
            payload.pop(name.removeprefix("without_"), None)
        messages[-1]["content"] = json.dumps(payload, ensure_ascii=False)
        yield name, messages, False
    messages = copy.deepcopy(original)
    messages[0]["content"] = ENTITIES
    yield "revised_prompt", copy.deepcopy(messages), False
    yield "revised_prompt_schema", messages, True
    yield "original_prompt_schema", copy.deepcopy(original), True
    messages = copy.deepcopy(original)
    messages[0]["content"] = ENTITIES
    payload = json.loads(messages[-1]["content"])
    payload.pop("question", None)
    messages[-1]["content"] = json.dumps(payload, ensure_ascii=False)
    yield "production_without_question", copy.deepcopy(messages), False
    yield "production_without_question_schema", messages, True


def relation_variants(event, entities):
    """Hold source, prompt and entity enum fixed; isolate the question field."""
    payload = json.loads(event["messages"][-1]["content"])
    payload.pop("existing_entities", None)
    payload["entities"] = entities
    for name in ("with_question", "without_question"):
        if name == "without_question":
            payload.pop("question", None)
        yield name, [{"role": "system", "content": RELATIONS},
                     {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]


async def probe_relations(helper, event, entities, audit):
    trials = []
    for name, messages in relation_variants(event, entities):
        request = {"model": helper.model, "messages": messages, "temperature": 0,
                   "max_tokens": helper.max_tokens, "chat_template_kwargs": {"enable_thinking": False},
                   "structured_outputs": {"disable_any_whitespace": True, "json": memory_output_schema("relations", entities=entities)}}
        trial = {"variant": name}
        try:
            response = await helper.client.post(helper.endpoint + "/v1/chat/completions", json=request)
            audit({"kind": "relation_replay", "variant": name, "request": request,
                   "status": response.status_code, "response": response.text})
            response.raise_for_status()
            choice = response.json()["choices"][0]
            trial["finish_reason"] = choice.get("finish_reason")
            if choice.get("finish_reason") == "length":
                raise ValueError("Relation extraction truncated")
            value = json.loads(choice["message"]["content"])
            validate_memory_output("relations", value)
            if any(row[i].strip() not in entities for row in value for i in (0, 2)):
                raise ValueError("Unverified relation endpoint")
            trial.update(relations=value, count=len(value))
        except Exception as exc:
            trial["error"] = f"{type(exc).__name__}: {exc}"
        trials.append(trial)
    return trials


def read_cases(source):
    root = Path(source)
    if (root / "evaluation").is_dir():
        root /= "evaluation"
    cases = []
    for path in sorted(root.glob("instances/*/attempt-*/trajectory.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                event = json.loads(line)
                if event.get("kind") == "memory_call" and event.get("operation") == "entities":
                    cases.append((str(path), event))
                    break
    if not cases:
        raise ValueError("No saved entities requests found; source must be a GRAM run/evaluation directory")
    return cases


async def probe(source, output, endpoint, model):
    cases = read_cases(source)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    audit_path = output / "helper-probe.jsonl"
    # Preserve evidence and refuse to append a different probe into an old report.
    with audit_path.open("x", encoding="utf-8") as audit_file:
        calls = []
        def audit(event):
            audit_file.write(json.dumps(event, ensure_ascii=False) + "\n")
            audit_file.flush()
            if event.get("kind") == "memory_call":
                calls.append({key: event.get(key) for key in ("operation", "response", "finish_reason", "usage")})
        helper = MemoryBackend(endpoint, model, audit=audit)
        report = {"model": model, "input_protocol": "gram-memory-source-facts-v1",
                  "cases": [], "controls": [], "checks_passed": False}
        try:
            for path, event in cases:
                case = {"source": path, "saved_response": event["response"], "trials": []}
                report["cases"].append(case)
                for name, messages, structured in variants(event):
                    request = {"model": model, "messages": messages, "temperature": 0,
                               "max_tokens": 2048, "chat_template_kwargs": {"enable_thinking": False}}
                    if structured:
                        request["structured_outputs"] = {"disable_any_whitespace": True, "json": memory_output_schema("entities")}
                    trial = {"variant": name}
                    try:
                        response = await helper.client.post(endpoint.rstrip("/") + "/v1/chat/completions", json=request)
                        audit({"kind": "replay", "source": path, "variant": name, "request": request,
                               "status": response.status_code, "response": response.text})
                        response.raise_for_status()
                        choice = response.json()["choices"][0]
                        trial["finish_reason"] = choice.get("finish_reason")
                        if choice.get("finish_reason") == "length":
                            raise ValueError("Entity extraction truncated")
                        value = json.loads(choice["message"]["content"])
                        validate_memory_output("entities", value)
                        trial.update(entities=value, count=len(value))
                    except Exception as exc:
                        trial["error"] = f"{type(exc).__name__}: {exc}"
                    case["trials"].append(trial)
                    print(json.dumps({"source": path, **trial}, ensure_ascii=False), flush=True)
                payload = json.loads(event["messages"][-1]["content"])
                graph = GraphMemory()
                audit({"kind": "production_edit_start", "source": path})
                first_call = len(calls)
                try:
                    edit = await helper.edit("memory_insert", payload["requested_facts"], payload["document"],
                                             payload.get("question", ""), graph, .9)
                    case.update(production_edit=edit, graph=graph.snapshot())
                except Exception as exc:
                    case["production_error"] = f"{type(exc).__name__}: {exc}"
                case["production_calls"] = calls[first_call:]
                # Use a completed extraction, never manually inject entity names.
                names = next((trial["entities"] for trial in reversed(case["trials"])
                              if trial.get("count") and "error" not in trial), [])
                if names:
                    names = list(dict.fromkeys(name.strip() for name in names))
                    case["relation_entities"] = names
                    audit({"kind": "relation_replay_start", "source": path})
                    case["relation_trials"] = await probe_relations(helper, event, names, audit)
            first_call = len(calls)
            report["controls"] = await check_helper_controls(helper)
            report["control_calls"] = calls[first_call:]
            report["checks_passed"] = all(c["passed"] for c in report["controls"]) and all(c.get("graph") for c in report["cases"])
        finally:
            await helper.aclose()
            (output / "helper-probe.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if report["checks_passed"] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    raise SystemExit(asyncio.run(probe(args.source, args.output, args.endpoint, args.model)))


if __name__ == "__main__":
    main()
