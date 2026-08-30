"""Dependency-free helpers for assistant-only ChatML loss masks."""

from collections.abc import Sequence


def validate_chatml_assistant_spans(source_spans: int, rendered_spans: int) -> None:
    """Validate rendered ChatML supervision without assuming 1:1 source turns.

    Qwen chat templates may intentionally omit historical assistant thinking
    turns while rendering a full conversation. The rendered token sequence is
    authoritative for the loss mask, so fewer rendered spans are valid. Zero
    supervision and extra rendered assistant headers remain errors.
    """
    if source_spans > 0 and rendered_spans == 0:
        raise ValueError("ChatML loss mask has no rendered assistant spans")
    if rendered_spans > source_spans:
        raise ValueError(
            f"ChatML rendered assistant span count exceeds source messages: {rendered_spans} > {source_spans}"
        )


def build_chatml_assistant_mask(
    input_ids: Sequence[int],
    *,
    im_start_id: int,
    im_end_id: int,
    assistant_header_ids: Sequence[int],
) -> tuple[list[int], int]:
    """Mark assistant payloads, including their closing ``<|im_end|>`` token."""
    ids = [int(token_id) for token_id in input_ids]
    header = [im_start_id, *(int(token_id) for token_id in assistant_header_ids)]
    if not assistant_header_ids:
        raise ValueError("assistant_header_ids must not be empty")

    mask = [0] * len(ids)
    spans = 0
    index = 0
    while index <= len(ids) - len(header):
        if ids[index : index + len(header)] != header:
            index += 1
            continue
        payload_start = index + len(header)
        try:
            payload_end = ids.index(im_end_id, payload_start)
        except ValueError as exc:
            raise ValueError("ChatML assistant span is missing its <|im_end|> token") from exc
        for token_index in range(payload_start, payload_end + 1):
            mask[token_index] = 1
        spans += 1
        index = payload_end + 1
    return mask, spans
