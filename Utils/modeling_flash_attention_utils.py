import inspect
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from import_utils import is_flash_attn_2_available, is_flash_attn_greater_or_equal

if is_flash_attn_2_available():
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input


def _get_unpad_data(attention_mask : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1,0))
    return (indices, cu_seqlens, max_seqlen_in_batch)

def _upad_input(
        query_layer : torch.Tensor,
        key_layer : torch.Tensor,
        value_layer : torch.Tensor,
        attention_mask : torch.Tensor,
        query_length : int
):
    indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
    batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape