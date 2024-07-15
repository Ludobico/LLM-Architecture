import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from Utils.Cache_utils import Cache, DynamicCache
from Utils.pytorch_utils import ALL_LAYERNORM_LAYERS
from llama_config import LlamaConfig

_CONFIG_FOR_DOC = "LlamaConfig"

def _get_unpad_data(attention_mask : torch.Tensor):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1,0))
    return (indices, cu_seqlens, max_seqlen_in_batch)

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states : torch.Tensor):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

ALL_LAYERNORM_LAYERS.append(LlamaRMSNorm)

class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings = 2048, base = 10000, device = None, scaling_factor = 1.0):
        super().__init__()
        self.scaling_facor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len_cached = max_position_embeddings
    
    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1,2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
    

class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    def forward(self, x, position_ids):
        position_ids = position_ids.float() / self.scaling_facor
        cos, sin = super().forward(x, position_ids)
        return cos, sin
    
class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    def forward(self, x, position_ids):
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_facor * seq_len / self.max_position_embeddings) - (self.scaling_facor - 1)
            ) ** (self.dim / (self.dim -2))
            inv_freq = 1.0 / (
                base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(x.device) / self.dim)
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        cos, sin = super().forward(x, position_ids)
        return cos, sin
    

def rotate_half(x : torch.Tensor):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=1)

def apply_rotary_pos_emb(q, k, cos, sin, postiion_ids = None, unsquzze_dim=1):
    cos = cos.unsqueeze(unsquzze_dim)
    sin = sin.unsqueeze(unsquzze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config =config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=self.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=self.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]
    
    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat([F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)]
            down_proj = sum(down_proj)
        
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        
        return down_proj
    

def repeat_kv(hidden_states : torch.Tensor, n_rep : int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:,:,None,:,:].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class LlamaAttention(nn.Module):
    def __init__(self, config : LlamaConfig, layer_idx : Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            raise ValueError(f"Instantiation {self.__class__.__name__} without passing a `layer_idx` is not recommended and will lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` When creating this class")
        
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size` : {self.hidden_size})"
                f" and `num_heads` : {self.num_heads}"
            )
