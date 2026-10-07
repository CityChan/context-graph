"""Keep actor backward decisions consistent across the training process group."""

import torch
import torch.distributed as dist


def require_finite_losses(**losses):
    """All ranks call once per microbatch, before any rank enters backward.

    The actor's default process group spans its FSDP/sequence-parallel ranks,
    just as the dynamic microbatch-count synchronization does. Stop the update
    on every rank if any loss is non-finite; zeroing a NaN loss is not safe.
    """
    bad = torch.stack([~torch.isfinite(loss.detach()).all() for loss in losses.values()]).to(torch.int32)
    if dist.is_initialized():
        dist.all_reduce(bad, op=dist.ReduceOp.MAX)
    failed = [name for name, flag in zip(losses, bad.tolist()) if flag]
    if failed:
        raise FloatingPointError(
            f"Non-finite actor loss on at least one rank: {', '.join(failed)}. "
            "Stopping all ranks before backward; inspect log-probs, advantages and KL."
        )
