import torch
import torch.nn as nn

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