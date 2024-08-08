from typing import Optional, Dict, Tuple, TYPE_CHECKING, Any, Callable, List, Union
import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import inspect

from ..Utils.Cache_utils import Cache, DynamicCache
from ..Utils.pytorch_utils import is_torch_greater_or_equal_than_2_4
from ..generation.configuration_utils import GenerationConfig

class GenerationMixin:
    def prepare_inputs_for_generaion(self, *args, **kwargs):
        raise NotImplementedError(
            "A model class needs to define a `prepare_inputs_for_generation` method in order to use `.generate()`"
        )
    
    def _prepare_model_inputs(self, inputs : Optional[torch.Tensor] = None, bos_token_id : Optional[torch.Tensor] = None, model_kwargs : Optional[Dict[str, torch.Tensor]] = None) -> Tuple[torch.Tensor, Optional[str], Dict[str, torch.Tensor]]:
        if(
            self.config.is_encoder_decoder and hasattr(self, 'encoder') and self.encoder.main_input_name != self.main_input_name
        ):
            input_name = self.encoder.main_input_name
        else:
            input_name = self.main_input_name

        
        model_kwargs = {k : v for k, v in model_kwargs.items() if v is not None or k != input_name}

        inputs_kwargs = model_kwargs.pop(input_name, None)
        if inputs_kwargs is not None and inputs is not None:
            raise ValueError(
                f"`inputs` : {inputs} were passed alongside {input_name} which is not allowed."
                f"Make sure to either pass {inputs} or {input_name}"
            )
        
        elif inputs_kwargs is not None:
            inputs = inputs_kwargs
        
        if input_name == "input_ids" and "inputs_embeds" in model_kwargs:
            if not self.config.is_encoder_decoder:
                has_inputs_embeds_forwarding = "inputs_embeds" in set(inspect.signature(self.prepare_inputs_for_generaion).parameters.keys())
                if not has_inputs_embeds_forwarding:
                    raise ValueError(
                        f"You passed `inputs_embeds` to `.generate()`, but the model class {self.__class__.__name__}."
                    )
            
            else:
                if inputs is not None:
                    raise ValueError("You passed `inputs_embeds` and `input_ids` to `.generate()`. plaese pick one")
            
            inputs, input_name = model_kwargs['inputs_embeds'], "inputs_embeds"

        
        inputs = self._maybe_initialize_input_ids_for_generation(inputs, bos_token_id, model_kwargs)
        return inputs, input_name, model_kwargs
    
    def _maybe_initialize_input_ids_for_generation(self, inputs : Optional[torch.Tensor] = None, bos_token_id : Optional[torch.Tensor] = None, model_kwargs : Optional[Dict[str, torch.Tensor]] = None) -> torch.LongTensor:
        if inputs is not None:
            return inputs
        
        encoder_outputs = model_kwargs.get("encoder_outputs")
        if self.config.is_encoder_decoder and encoder_outputs is not None:
            shape = encoder_outputs.last_hidden_state.size()[:-1]
            return torch.ones(shape, dtype=torch.long, device=self.device) * -100
        
        batch_size = 1

        for value in model_kwargs.values():
            if isinstance(value, torch.Tensor):
                batch_size = value.shape[0]
                break

        if "inputs_embeds" in model_kwargs:
            return torch.ones((batch_size, 0), dtype=torch.long, device=self.device)
        
        if bos_token_id is None:
            raise ValueError("bos_token_ids has to be defined when no `input_ids` are provided")
        
        return torch.ones((batch_size, 1), dtype=torch.long, device=self.device) * bos_token_id
    
    def _prepare_attention_mask_for_generation(
            self, inputs : torch.Tensor, pad_token_id : Optional[torch.Tensor], eos_token_id : Optional[torch.Tensor]
    ) -> torch.LongTensor:
        default_attention_mask = torch.ones(inputs.shape[:2], dtype=torch.long, device=inputs.device)

        if pad_token_id is None:
            return default_attention_mask
        
        is_input_ids = len(inputs.shape) == 2 and inputs.dtype in [torch.int, torch.long]
        if not is_input_ids:
            return default_attention_mask
        
        if inputs.device.type == 'mps' and not is_torch_greater_or_equal_than_2_4:
            raise ValueError(
                "Can't infer missing attention mask on `mps` device for torch<2.4."
            )
        
        is_pad_token_in_inputs = (pad_token_id is not None) and (torch.isin(elements=inputs, test_elements=pad_token_id).any())
        is_pad_token_not_equal_to_eos_token_id = (eos_token_id is None) or ~(torch.isin(elements=eos_token_id, test_elements=pad_token_id).any())
        can_infer_attention_mask = is_pad_token_in_inputs * is_pad_token_not_equal_to_eos_token_id
        attention_mask_from_padding = inputs.ne(pad_token_id).long()

        attention_mask = (attention_mask_from_padding * can_infer_attention_mask + default_attention_mask * ~can_infer_attention_mask)
        return attention_mask
    
    def _prepare_encoder_decoder_kwargs_for_generation(
            self, inpus_tensor : torch.Tensor, model_kwargs, model_input_name : Optional[str], generation_config : GenerationConfig
    ) -> Dict[str, Any]:
        pass