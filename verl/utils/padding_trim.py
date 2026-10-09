"""Opt-in rectangular text-only padding trim and allocator maintenance."""
import os

import torch


def trim_micro_batch(batch, bucket):
    """Only remove columns masked on EVERY row; preserve response alignment.

    Return (inputs, original response width). The caller pads output logprobs
    back to that width before loss calculation; new positions never enter loss.
    """
    response_width = batch['responses'].shape[-1]
    if bucket <= 0 or batch.get('multi_modal_inputs') is not None:
        return batch, response_width
    mask = batch['attention_mask']
    width = mask.shape[-1]
    prompt_width = width - response_width
    if prompt_width < 1:
        return batch, response_width
    occupied = mask.bool().any(0).nonzero().flatten()
    first = min(int(occupied[0]) if occupied.numel() else prompt_width - 1, prompt_width - 1)
    end = max(int(occupied[-1]) + 1 if occupied.numel() else prompt_width + 1, prompt_width + 1)
    target = min(width, ((end - first + bucket - 1) // bucket) * bucket)
    left = max(0, end - target)
    right = min(width, left + target)
    trimmed = dict(batch)
    for key in ('input_ids', 'attention_mask', 'position_ids'):
        trimmed[key] = batch[key][..., left:right]
    trimmed['responses'] = batch['responses'][..., :right - prompt_width]
    return trimmed, response_width


def release_fragmented_cache():
    threshold = float(os.getenv('VERL_FRAG_EMPTY_CACHE_GB', '0'))
    if threshold > 0 and torch.cuda.is_available():
        if torch.cuda.memory_reserved() - torch.cuda.memory_allocated() > threshold * 1024**3:
            torch.cuda.empty_cache()


def reset_peak_memory():
    if (int(os.getenv('VERL_PAD_TRIM_BUCKET', '0')) > 0 or
            float(os.getenv('VERL_FRAG_EMPTY_CACHE_GB', '0')) > 0) and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
