"""Assistant-only token masks for chat templates without generation blocks."""

from __future__ import annotations

import re
from typing import Any


def chat_template_with_generation_blocks(template: Any) -> Any:
    """Wrap a Qwen-style assistant branch in a Jinja generation block.

    The generation extension records token spans but emits no text, so the
    patched template preserves model inputs while enabling Transformers'
    native assistant mask.
    """
    if not isinstance(template, str) or not template:
        return template
    if re.search(r"{%-?\s*generation\s*-?%}", template):
        return template

    assistant_branch = re.search(
        r"{%-?\s*elif\s+message\.role\s*==\s*['\"]assistant['\"]\s*-?%}",
        template,
    )
    if assistant_branch is None:
        return template
    following_branch = re.search(
        r"{%-?\s*elif\s+message\.role\s*==\s*['\"]tool['\"]\s*-?%}",
        template[assistant_branch.end() :],
    )
    if following_branch is None:
        return template

    following_start = assistant_branch.end() + following_branch.start()
    return (
        template[: assistant_branch.end()]
        + "{%- generation %}"
        + template[assistant_branch.end() : following_start]
        + "{%- endgeneration %}"
        + template[following_start:]
    )


def assistant_token_mask_from_offsets(
    *,
    tokenizer,
    processor,
    messages: list[dict[str, Any]],
    tools,
    apply_chat_template_kwargs: dict[str, Any],
    input_ids: Any,
    attention_mask: Any,
) -> Any:
    """Recover assistant content spans from one complete rendered conversation.

    Some modern chat templates do not contain Jinja ``generation`` blocks and
    are not prefix-stable when rendered one message at a time. Insert temporary
    text markers around assistant content, render the complete conversation,
    remove the markers, and map the resulting character spans through a fast
    tokenizer's offset mapping. The markers never enter the returned token IDs.
    """
    import torch

    render_kwargs = {
        "tools": tools,
        "add_generation_prompt": False,
        "tokenize": False,
        **apply_chat_template_kwargs,
    }
    rendered = processor.apply_chat_template(messages, **render_kwargs)
    if not isinstance(rendered, str):
        raise ValueError("Chat template did not return rendered text for assistant-mask fallback.")

    marker_stem = "<|contextgraph_assistant_mask_marker|>"
    while marker_stem in rendered:
        marker_stem += "_"

    marked_messages: list[dict[str, Any]] = []
    markers: list[tuple[str, str]] = []
    for index, message in enumerate(messages):
        marked = dict(message)
        if message.get("role") == "assistant":
            content = message.get("content", "")
            if not isinstance(content, str):
                raise ValueError(
                    "Assistant-mask offset fallback currently requires text-only assistant content."
                )
            start_marker = f"{marker_stem}{index}:start"
            end_marker = f"{marker_stem}{index}:end"
            if start_marker in rendered or end_marker in rendered:
                raise ValueError("Assistant-mask marker collides with rendered conversation text.")
            marked["content"] = f"{start_marker}{content}{end_marker}"
            markers.append((start_marker, end_marker))
        marked_messages.append(marked)

    if not markers:
        return torch.zeros_like(attention_mask)

    marked_rendered = processor.apply_chat_template(marked_messages, **render_kwargs)
    if not isinstance(marked_rendered, str):
        raise ValueError("Marked chat template did not return rendered text.")

    clean_parts: list[str] = []
    assistant_spans: list[tuple[int, int]] = []
    cursor = 0
    clean_length = 0
    for start_marker, end_marker in markers:
        start = marked_rendered.find(start_marker, cursor)
        if start < 0:
            raise ValueError("Chat template removed an assistant-mask start marker.")
        before = marked_rendered[cursor:start]
        clean_parts.append(before)
        clean_length += len(before)

        content_start = start + len(start_marker)
        end = marked_rendered.find(end_marker, content_start)
        if end < 0:
            raise ValueError("Chat template removed an assistant-mask end marker.")
        assistant_content = marked_rendered[content_start:end]
        span_start = clean_length
        clean_parts.append(assistant_content)
        clean_length += len(assistant_content)
        assistant_spans.append((span_start, clean_length))
        cursor = end + len(end_marker)

    clean_parts.append(marked_rendered[cursor:])
    clean_rendered = "".join(clean_parts)
    if clean_rendered != rendered:
        raise ValueError(
            "Removing assistant-mask markers did not reproduce the original rendered conversation."
        )

    try:
        encoded = tokenizer(
            rendered,
            add_special_tokens=False,
            return_attention_mask=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
    except (TypeError, NotImplementedError) as exc:
        raise ValueError(
            "Assistant-mask fallback requires a fast tokenizer with offset mappings."
        ) from exc

    offset_input_ids = torch.as_tensor(encoded["input_ids"])
    offsets = torch.as_tensor(encoded["offset_mapping"])
    if offset_input_ids.ndim == 2:
        offset_input_ids = offset_input_ids[0]
    if offsets.ndim == 3:
        offsets = offsets[0]
    if offset_input_ids.shape != input_ids.shape or not torch.equal(
        offset_input_ids.to(input_ids.device), input_ids
    ):
        raise ValueError(
            "Rendered-text tokenization does not match apply_chat_template tokenization; "
            "cannot construct an exact assistant mask."
        )

    loss_mask = torch.zeros_like(attention_mask)
    for token_index, (token_start, token_end) in enumerate(offsets.tolist()):
        if token_end <= token_start:
            continue
        if any(token_start < span_end and token_end > span_start for span_start, span_end in assistant_spans):
            loss_mask[token_index] = 1
    loss_mask *= attention_mask
    if not bool(loss_mask.any()):
        raise ValueError("Offset mapping produced an empty assistant token mask.")
    return loss_mask
