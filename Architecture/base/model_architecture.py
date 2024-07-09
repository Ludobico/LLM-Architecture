from transformers import AutoModelForCausalLM
from torchinfo import summary
import torch
from typing import Optional

class ModelArchitecture:
  def __init__(self, model_id : str, cache_dir : Optional[str] = None) -> None:
    self.model_id = model_id
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if cache_dir is not None:
      self.huggingface_cache_dir = cache_dir
  
  def get_model_architecture(self) -> None:
    model = AutoModelForCausalLM.from_pretrained(self.model_id, cache_dir = self.huggingface_cache_dir, device = self.device)
    print(model)
  
  def get_model_summary(self) -> None:
    model = AutoModelForCausalLM.from_pretrained(self.model_id, cache_dir = self.huggingface_cache_dir, device = self.device)
    summary(model)