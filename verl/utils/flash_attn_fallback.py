"""PyTorch fallback for flash_attn.bert_padding functions."""
import torch
from einops import rearrange

def index_first_axis(values, indices):
    return torch.index_select(values, 0, indices)

def unpad_input(hidden_states, attention_mask, **kwargs):
    seqlens = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen = int(seqlens.max().item())
    cu_seqlens = torch.zeros(seqlens.shape[0] + 1, dtype=torch.int32, device=seqlens.device)
    cu_seqlens[1:] = torch.cumsum(seqlens, dim=0)
    hidden_states_flat = hidden_states.reshape(-1, *hidden_states.shape[2:])
    hidden_states_unpad = hidden_states_flat[indices]
    return hidden_states_unpad, indices, cu_seqlens, max_seqlen

def pad_input(**kwargs):
    hidden_states_unpad = kwargs.get("hidden_states_unpad") or kwargs.get("hidden_states")
    indices = kwargs["indices"]
    batch = kwargs.get("batch") or kwargs.get("batch_size")
    seqlen = kwargs["seqlen"]
    output = torch.zeros(
        batch * seqlen, *hidden_states_unpad.shape[1:],
        dtype=hidden_states_unpad.dtype, device=hidden_states_unpad.device)
    output[indices] = hidden_states_unpad
    return output.reshape(batch, seqlen, *hidden_states_unpad.shape[1:])
