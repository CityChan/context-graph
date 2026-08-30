"""Dependency-free helpers for assistant-only ChatML loss masks."""

from collections.abc import Sequence


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
