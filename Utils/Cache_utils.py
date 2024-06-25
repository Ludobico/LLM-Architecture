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
