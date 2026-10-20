"""Loss masks over sampled token prefixes, without re-rendering changed history.

Used by Route A offline SFT. This does not enable training-time graph rewrites
for online RL, whose policy-logprob and credit alignment have a separate contract.
"""
import copy
import hashlib
import json


def checksum(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def tokenizer_signature(tokenizer):
    return checksum({"vocab": tokenizer.get_vocab(),
                     "special_tokens": tokenizer.special_tokens_map,
                     "chat_template": tokenizer.chat_template})


def token_list(value):
    if not isinstance(value, list) or not value or any(type(i) is not int or i < 0 for i in value):
        raise ValueError("Expected nonempty native token ID list")
    return value


def plain_messages(messages):
    return [{k: v for k, v in m.items() if k != "train"} for m in messages]


def segments_from_requests(requests, max_length, signature):
    """Merge only identical token AND message prefixes, within one context UID.

    Each sampled response is supervised once. At a rewrite/reset the old messages
    become conditioning only. Input headers, observations and unsampled tokens
    always have zero loss, including any synthesized assistant history.
    """
    segments, active = [], {}
    for index, record in enumerate(requests):
        prompt = token_list(record.get("input_ids"))
        output = token_list(record.get("output_ids"))
        if len(prompt) + len(output) > max_length:
            raise ValueError("Overlength sampled request; refusing input truncation")
        uid = record.get("context_uid")
        messages = copy.deepcopy(record.get("messages"))
        if not uid or not messages or not isinstance(record.get("response"), str):
            raise ValueError("Missing sampled context identity/messages/response")
        previous = active.get(uid)
        compatible = (previous is not None
                      and prompt[:len(previous["agent_context"]["input_ids"])] == previous["agent_context"]["input_ids"]
                      and plain_messages(messages[:len(previous["messages"])]) == plain_messages(previous["messages"]))
        if not compatible:
            previous = {"messages": [], "agent_context": {"version": 1, "input_ids": [],
                        "loss_mask": [], "spans": []}, "tokenizer_sha256": signature}
            segments.append(previous)
            active[uid] = previous
        context = previous["agent_context"]
        old_length = len(context["input_ids"])
        context["input_ids"] = prompt + output
        context["loss_mask"] += [0] * (len(prompt) - old_length) + [1] * len(output)
        context["spans"].append({"start": len(prompt), "end": len(prompt) + len(output),
                                 "prefix_sha256": checksum(prompt), "output_sha256": checksum(output),
                                 "request_index": index})
        old_messages = len(previous["messages"])
        previous["messages"] += [dict(m, train=False) for m in messages[old_messages:]]
        previous["messages"].append({"role": "assistant", "content": record["response"], "train": True})
        context["sha256"] = checksum({k: v for k, v in context.items() if k != "sha256"})
    return segments


def validate_agent_context(row, max_length, signature):
    context = row.get("agent_context")
    if not isinstance(context, dict) or context.get("version") != 1:
        raise ValueError("Missing/unsupported agent_context")
    if row.get("tokenizer_sha256") != signature:
        raise ValueError("agent_context tokenizer mismatch")
    if checksum({k: v for k, v in context.items() if k != "sha256"}) != context.get("sha256"):
        raise ValueError("agent_context checksum mismatch")
    ids = token_list(context.get("input_ids"))
    if len(ids) > max_length:
        raise ValueError("agent_context exceeds max_length; truncation is forbidden")
    expected = [0] * len(ids)
    last_end = 0
    spans = context.get("spans", [])
    for span in spans:
        start, end = span["start"], span["end"]
        if not (type(start) is int and type(end) is int and last_end <= start < end <= len(ids)):
            raise ValueError("Invalid/overlapping supervised spans")
        if checksum(ids[:start]) != span["prefix_sha256"] or checksum(ids[start:end]) != span["output_sha256"]:
            raise ValueError("Sampled conditioning or output changed")
        expected[start:end] = [1] * (end - start)
        last_end = end
    if not spans or expected != context.get("loss_mask"):
        raise ValueError("agent_context loss mask must cover exactly sampled outputs")
    targets = [m for m in row["messages"] if m.get("train") is True]
    if len(targets) != len(spans) or any(m["role"] != "assistant" for m in targets):
        raise ValueError("Message train flags do not match sampled responses")
    return ids, expected
