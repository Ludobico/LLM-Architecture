import torch
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

@dataclass
class Cache:
  def update(self, key_states : torch.Tensor, value_states : torch.Tensor, layer_idx : int, cache_kwargs : Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    raise NotImplementedError("Make sure to implement `update` in a subclass.")
  
  def get_seq_length(self, layer_idx : Optional[int] = 0) -> int:
    raise NotImplementedError("Make sure to implement `get_seq_length` in a subclass.")
  
  def get_max_length(self) -> Optional[int]:
    raise NotImplementedError("Make sure to implement `get_max_length` in a subclass.")
  
  def get_usable_length(self, new_seq_length : int, layer_idx : Optional[int] = 0) -> int:
    max_length = self.get_max_length()
    previous_seq_length = self.get_seq_length(layer_idx)
    if max_length is not None and previous_seq_length + new_seq_length > max_length:
      return max_length - new_seq_length
    return previous_seq_length

class DynamicCache(Cache):
  def __init__(self) -> None:
    self.key_cache : List[torch.Tensor] = []
    self.value_cache : List[torch.Tensor] = []
    self._seen_tokens = 0
  
  def __getitem__(self, layer_idx : int) -> List[Tuple[torch.Tensor]]:
    if layer_idx < len(self):
      return (self.key_cache[layer_idx], self.value_cache[layer_idx])
    else:
      raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")
  
  def __iter__(self):
    for layer_idx in range(len(self)):
      yield (self.key_cache[layer_idx], self.value_cache[layer_idx])
  
  def __len__(self):
    return len(self.key_cache)
  
  def update(
      self,
      key_states : torch.Tensor,
      value_states : torch.Tensor,
      layer_idx : int,
      cache_kwargs : Optional[Dict[str, Any]] = None,
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    if layer_idx == 0:
      self._seen_tokens += key_states.shape[-2]

    if len(self.key_cache) <= layer_idx:
      self.key_cache.append(key_states)
      self.value_cache.append(value_states)
    
    else:
      self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
      self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
    
    return self.key_cache[layer_idx], self.value_cache[layer_idx]
  
  def get_seq_length(self, layer_idx : Optional[int] = 0) -> int:
    if len(self.key_cache) <= layer_idx:
      return 0
    return self.key_cache[layer_idx].shape[-2]
  
  def get_max_length(self) -> Optional[int]:
    return None
  
  def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
    legacy_cache = ()
    for layer_idx in range(len(self)):
      legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx]),)
    return legacy_cache
  
  @classmethod
  def from_legacy_cache(cls, past_key_values : Optional[Tuple[Tuple[torch.FloatTensor]]] = None) -> "DynamicCache":
    cache = cls()
    if past_key_values is not None:
      for layer_idx in range(len(past_key_values)):
        key_states, value_states = past_key_values[layer_idx]
        cache.update(key_states, value_states, layer_idx)
    return cache
  
  def crop(self, max_length : int):
    if max_length < 0:
      max_length = self.get_seq_length() - abs(max_length)

    if self.get_seq_length() <= max_length:
      return
    
    self._seen_tokens = max_length

    for idx in range(len(self.key_cache)):
      self.key_cache[idx] = self.key_cache[idx][..., :max_length, :]
      self.value_cache[idx] = self.value_cache[idx][..., :max_length, :]
  
  def batch_split(self, full_batch_size : int, split_size : int) -> List["DynamicCache"]:
    out = []
    for i in range(0, full_batch_size, split_size):
      current_split = DynamicCache()
      current_split._seen_tokens = self._seen_tokens
      current_split.key_cache = [tensor[i : i + split_size] for tensor in self.key_cache]
      current_split.value_cache = [tensor[i : i + split_size] for tensor in self.value_cache]
      out.append(current_split)
    return out
  
  @classmethod
  def from_batch_splits(cls, splits: List["DynamicCache"]) -> "DynamicCache":
      """This is the opposite of the above `batch_split()` method. This will be used by `stack_model_outputs` in
      `generation.utils`"""
      cache = cls()
      for idx in range(len(splits[0])):
          layer_keys = torch.cat([current.key_cache[idx] for current in splits], dim=0)
          layer_values = torch.cat([current.value_cache[idx] for current in splits], dim=0)
          cache.update(layer_keys, layer_values, idx)
      return cache

  def batch_repeat_interleave(self, repeats: int):
      """Repeat the cache `repeats` times in the batch dimension. Used in contrastive search."""
      for layer_idx in range(len(self)):
          self.key_cache[layer_idx] = self.key_cache[layer_idx].repeat_interleave(repeats, dim=0)
          self.value_cache[layer_idx] = self.value_cache[layer_idx].repeat_interleave(repeats, dim=0)

  def batch_select_indices(self, indices: torch.Tensor):
      """Only keep the `indices` in the batch dimension of the cache. Used in contrastive search."""
      for layer_idx in range(len(self)):
          self.key_cache[layer_idx] = self.key_cache[layer_idx][indices, ...]
          self.value_cache[layer_idx] = self.value_cache[layer_idx][indices, ...]

