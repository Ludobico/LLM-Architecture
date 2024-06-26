import os, sys, math
sys.path.append(os.path.dirname(os.path.join(__file__, '..')))
import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Union, Any
from Utils.Cache_utils import Cache


class Qwen2Config:
  model_type = "qwen2"
  keys_to_ignore_at_inference = ["past_key_values"]

  def __init__( self,
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=22016,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=28,
        attention_dropout=0.0,
        **kwargs,):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        self.max_window_layers = max_window_layers

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

class Qwen2MLP(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.hidden_size = config.hidden_size
    self.intermediate_size = config.intermediate_size
    self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
    self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
    self.down_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
    # activation function
    self.act_fn = nn.SiLU()
  
  def forward(self, hidden_state):
    return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))

class Qwen2RMSNorm(nn.Module):
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

class Qwen2RotaryEmbedding(nn.Module):
  def __init__(self, dim, max_position_embeddings = 2048, base = 10000, device = None):
    super().__init__()

    self.dim = dim
    self.max_position_embeddings = max_position_embeddings
    self.base = base
    inv_freq = 1.0 / (self.base **(torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device)))
    # obsidian
    self.register_buffer("inv_freq", inv_freq, persistent=False)
    self._set_cos_sin_cache(max_position_embeddings, inv_freq.device, torch.int64)
  
  def _set_cos_sin_cache(self, seq_len, device, dtype):
    self.max_seq_len_cached = seq_len
    t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)

    # obsidian
    freqs = torch.outer(t, self.inv_freq)
    emb = torch.cat((freqs, freqs), dim=1)
    self.register_buffer("cos_cached", emb.cos().to(dtype=dtype), persistent=False)
    self.register_buffer("sin_cached", emb.sin().to(dtype=dtype), persistent=False)

  def forward(self, x : torch.Tensor, seq_len = None):
    if seq_len > self.max_seq_len_cached:
      self._set_cos_sin_cache(seq_len, x.device, x.dtype)

    
    return (
      self.cos_cached[:seq_len].to(dtype=x.dtype),
      self.sin_cached[:seq_len].to(dtype=x.dtype)
    )

def rotate_half(x : torch.Tensor):
  x1 = x[..., : x.shape[-1] // 2]
  x2 = x[..., x.shape[-1] // 2 :]
  return torch.cat((-x2, x1), dim=1)
def apply_rotary_pos_emb(q, k, cos : torch.Tensor, sin : torch.Tensor, position_ids, unsqueeze_dim = 1):
  cos = cos[position_ids].unsqueeze(unsqueeze_dim)
  sin = sin[position_ids].unsqueeze(unsqueeze_dim)
  q_embed = ( q * cos) + (rotate_half(q) * sin)
  k_embed = ( k * cos) + (rotate_half(k) * sin)
  return q_embed, k_embed

def repeat_kv(hidden_states : torch.Tensor, n_rep : int) -> torch.Tensor:
  batch, num_key_value_heads, slen, head_dim = hidden_states.shape
  if n_rep == 1:
    return hidden_states
  
  hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
  return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class Qwen2Attention(nn.Module):
  def __init__(self, config : Qwen2Config, layer_idx : Optional[int] = None):
    super().__init__()
    self.config = config
    self.layer_idx = layer_idx
    if layer_idx is None:
      raise ValueError("layer_idx cannot be None")
    self.hidden_size = config.hidden_size
    self.num_heads = config.num_attention_heads
    self.head_dim = self.hidden_size // self.num_heads
    self.num_key_value_heads = config.num_key_value_heads
    self.num_key_value_groups = self.num_heads // self.num_key_value_heads
    self.max_position_embeddings = config.max_position_embeddings
    self.rope_theta = config.rope_theta
    self.is_causal = True
    self.attention_dropout = config.attention_dropout

    self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
    self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
    self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
    self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    self.rotary_emb = Qwen2RotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings, base=self.rope_theta)

  def forward(self, hidden_states : torch.Tensor, attention_mask : Optional[torch.Tensor] = None, position_ids : Optional[torch.LongTensor] = None, past_key_value : Optional[Cache] = None, output_attentions : bool = False, use_cache : bool = False, cache_position : Optional[torch.LongTensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
      if self.layer_idx is None:
        raise ValueError("for auto-regressive decoding with k/v caching, please make sure to initialize the attention class for a layer index")
    
      kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if past_key_value is not None:
      cache_kwargs = {"sin" : sin, "cos" : cos, "cache_position" : cache_position}
      key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
      raise ValueError(
        f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is {attn_weights.size()}"
      )
    
    if attention_mask is not None:
      causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
      attn_weights = attn_weights + causal_mask
    
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.device)
    attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
      raise ValueError(
        f"Attention output should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is {attn_output.size()}")

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
      attn_weights = None
    
    return attn_output, attn_weights, past_key_value



class Qwen2SdpaAttention(Qwen2Attention):
  """
  This Module inherits from Qwen2Attention and thus uses the `__init__` method of the parent class.
  """

  def forward(self, hidden_states : torch.Tensor, attention_mask : Optional[torch.Tensor] = None, position_ids : Optional[torch.LongTensor] = None, past_key_value : Optional[Cache] = None, output_attentions : bool = False, use_cache : bool = False, cache_position : Optional[torch.LongTensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if output_attentions :
      return super().forward(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache
      )
  
    bsz, q_len = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
      kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if past_key_value is not None:
      cache_kwargs = {"sin" : sin, "cos" : cos, "cache_position" : cache_position}
      key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
    
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    causal_mask = attention_mask
    if attention_mask is not None:
      causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
    
    if query_states.device.type == "cuda" and attention_mask is not None:
      query_states = query_states.contiguous()
      key_states = key_states.contiguous()
      value_states = value_states.contiguous()
    
    is_causal = True if causal_mask is None and q_len > 1 else False

    attn_output = torch.nn.functional.scaled_dot_product_attention(
      query_states, key_states, value_states, attn_mask=causal_mask, dropout_p=self.attention_dropout if self.training else 0.0,
      is_causal=is_causal
    )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value

QWEN2_ATTENTION_CLASSES = {
  "eager" : Qwen2Attention,
  "sdpa" : Qwen2SdpaAttention
}
  

class Qwen2DecoderLayer(nn.Module):
  def __init__(self, config : Qwen2Config, layer_idx : int):
    super().__init__()
    self.hidden_size = config.hidden_size

    self.self_attn = QWEN2_ATTENTION_CLASSES["sdpa"](config, layer_idx)

    self.mlp = Qwen2MLP(config)
    self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

  def forward(
      self,
      hidden_states : torch.Tensor,
      attention_mask : Optional[torch.Tensor] = None,
      position_ids : Optional[torch.LongTensor] = None,
      past_key_value : Optional[Tuple[torch.Tensor]] = None,
      output_attentions : Optional[bool] = False,
      use_cache : Optional[bool] = False,
      cache_position : Optional[torch.LongTensor] = None,
      **kwargs,
  ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)

    hidden_states, self_attn_weights, present_key_value = self.self_attn(
      hidden_states=hidden_states,
      attention_mask=attention_mask,
      position_ids=position_ids,
      past_key_value=past_key_value,
      output_attentions=output_attentions,
      use_cache=use_cache,
      cache_position=cache_position,
    )

    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    if output_attentions:
      outputs += (self_attn_weights,)
    
    if use_cache:
      outputs += (present_key_value,)
    
    return outputs
  
  






    
