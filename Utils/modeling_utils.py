import collections, sys, os
sys.path.append(os.path.dirname(os.path.realpath(os.path.join(__file__, '..'))))
import copy
import functools
import gc
import importlib.metadata
import inspect
import itertools
import json
import re
import shutil
import tempfile
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial, wraps
from threading import Thread
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from zipfile import is_zipfile

import torch
from packaging import version
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss, Identity
from torch.utils.checkpoint import checkpoint

from Utils.import_utils import is_bitsandbytes_available, ENV_VARS_TRUE_VALUES, is_torch_xla_available
from generation.utils import GenerationMixin
from generation.configuration_utils import GenerationConfig
from intergrations.peft import PeftAdapterMixin
from intergrations.deepspeed import is_deepspeed_available, is_deepspeed_zero3_enabled
from Utils.hub import PushToHubMixin
from Utils import DUMMY_INPUTS
from configuration_utils import PretrainedConfig

XLA_USE_BF16 = os.environ.get("XLA_USE_BF16", "0").upper()
XLA_DOWNCAST_BF16 = os.environ.get("XLA_DOWNCAST_BF16", "0").upper()
PARAM_RENAME_WARNING = "WARNING: The parameter name in your model is different from its name in the checkpoint. "

def get_parameter_dtype(parameter : Union[nn.Module, GenerationMixin, "ModuleUtilsMixin"]):
    last_dtype = None

    for t in parameter.parameters():
        last_dtype = t.dtype
        if t.is_floating_point():
            if XLA_USE_BF16 in ENV_VARS_TRUE_VALUES and is_torch_xla_available():
                return torch.bfloat16
            if XLA_DOWNCAST_BF16 in ENV_VARS_TRUE_VALUES and is_torch_xla_available():
                if t.dtype == torch.float:
                    return torch.bfloat16
                if t.dtype == torch.double:
                    return torch.float32
            return t.dtype
        
    if last_dtype is not None:
        return last_dtype
    
    def find_tensor_attributes(module : nn.Module) -> List[Tuple[str, torch.Tensor]]:
        tuples = [(k,v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
        return tuples
    
    gen = parameter._named_members(get_members_fn=find_tensor_attributes)
    last_tuple = None

    for tuple in gen:
        last_tuple = tuple
        if tuple[1].is_floating_point():
            return tuple[1].dtype
        
    if last_tuple is not None:
        return last_tuple[1].dtype
    
    for t in parameter.buffers():
        last_dtype = t.dtype
        if t.is_floating_point():
            return t.dtype
    return last_dtype

class ModuleUtilsMixin:
    @staticmethod
    def _hook_rss_memory_pre_forward(module, *args, **kwargs):
        try:
            import psutil
        except ImportError:
            raise ImportError("You need to install psutil (pip install psutil) to use memory tracing.")
        
        process = psutil.Process(os.getpid())
        mem = process.memory_info()
        module.mem_rss_pre_forward = mem.rss
        return None
    
    @staticmethod
    def _hook_rss_memory_post_forward(module, *args, **kwargs):
        try:
            import psutil
        except ImportError:
            raise ImportError('You need to instal psutil to se memory tracing')
        
        process = psutil.Process(os.getpid())
        mem = process.memory_info()
        module.mem_rss_post_forward = mem.rss
        mem_rss_diff = module.mem_rss_post_forward - module.mem_rss_pre_forward
        module.mem_rss_diff = mem_rss_diff + (module.mem_rss_diff if hasattr(module, "mem_rss_diff") else 0)
        return None
    
    def add_memory_hooks(self):
        for module in self.modules():
            module.register_forward_pre_hook(self._hook_rss_memory_post_forward)
            module.register_forward_hook(self._hook_rss_memory_pre_forward)
        self.reset_memory_hooks_state()
    
    def reset_memory_hooks_state(self):
        for module in self.modules():
            module.mem_rss_diff = 0
            module.mem_rss_post_forward = 0
            module.mem_rss_pre_forward = 0
    
    @property
    def device(self) -> torch.device:
        return get_parameter_dtype(self)
    
    @property
    def dtype(self) -> torch.dtype:
        return get_parameter_dtype(self)
    
    def invert_attention_mask(self, encoder_attention_mask : torch.Tensor) -> torch.Tensor:
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        if encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
        
        encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=self.dtype)
        encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * torch.finfo(self.dtype).min

    
    @staticmethod
    def create_extended_attention_mask_for_decoder(input_shape, attention_mask, device=None):
        if device is not None:
            warnings.warn("The `device` argument is deprecated and will be removed in v5 of Transformers.", FutureWarning)
        
        else:
            device = attention_mask.device
        
        batch_size, seq_length = input_shape
        seq_ids = torch.arange(seq_length, device=device)
        causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
        causal_mask = causal_mask.to(attention_mask.dtype)

        if causal_mask.shape[1] < attention_mask.shape[1]:
            prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1]
            causal_mask = torch.cat(
                [
                    torch.ones((batch_size, seq_length, prefix_seq_len), device=device, dtype=causal_mask.dtype)
                ],
                axis=1
            )
        
        extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
        return extended_attention_mask
    
    def get_extended_attention_mask(self, attention_mask : torch.Tensor, input_shape : Tuple[int], device : torch.device = None, dtype : torch.float = None) -> torch.Tensor:
        if dtype is None:
            dtype =  self.dtype

        if not (attention_mask.dim() == 2 and self.config.is_decoder):
            if device is not None:
                warnings.warn(
                    "The `device` argument is deprecated and will be removed in v5 of Transformers.", FutureWarning
                )
        
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, : ,:]
        elif attention_mask.dim() == 2:
            if self.config.is_decoder:
                extended_attention_mask = ModuleUtilsMixin.create_extended_attention_mask_for_decoder(input_shape, attention_mask, device)
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(f"Wrong shape for an input with shape {input_shape}")
        

    def get_head_mask(self, head_mask : Optional[torch.Tensor], num_hidden_layers : int, is_attention_chunked : bool = False) -> torch.Tensor:
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
            if is_attention_chunked is True:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers
        
        return head_mask
    
    def _convert_head_mask_to_5d(self, head_mask : torch.Tensor, num_hidden_layers):
        if head_mask.dim() == 1:
            # torch.Size([3, 4]) -> torch.Size([1, 1, 3, 4, 1, 1])
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        assert head_mask.dim() == 5, f"head_mask.dim != 5, {head_mask.dim()}"
        head_mask = head_mask.to(dtype=self.dtype)
        return head_mask
    
    def num_parameters(self, only_trainable : bool = False, exclude_embeddings : bool = False) -> int:
        if exclude_embeddings:
            embedding_param_names = [
                f"{name}.weight" for name, module_type in self.named_modules() if isinstance(module_type, nn.Embedding)
            ]
            total_parameters = [
                parameter for name, parameter in self.named_parameters() if name not in embedding_param_names
            ]
        else:
            total_parameters = list(self.parameters())
        
        total_numel = []
        is_loaded_in_4bit = getattr(self, "is_loaded_in_4bit", False)

        if is_loaded_in_4bit:
            if is_bitsandbytes_available():
                import bitsandbytes as bnb
            else:
                raise ValueError("Please install bitsandbytes to use 4-bit weights.")
            
        for param in total_parameters:
            if param.requires_grad or not only_trainable:
                if is_loaded_in_4bit and isinstance(param, bnb.nn.Params4bit):
                    if hasattr(param, 'element_size'):
                        num_bytes = param.element_size()
                    elif hasattr(param, 'quant_storage'):
                        num_bytes = param.quant_storage.itemsize()
                    else:
                        num_bytes = 1
                    total_numel.append(param.numel() * 2 * num_bytes)
            else:
                total_numel.append(param.numel())
        return sum(total_numel)
    
    def estimate_tokens(self, input_dict : Dict[str, Union[torch.Tensor, Any]]) -> int:
        if not hasattr(self, 'warnings_issued'):
            self.warnings_issued = {}
        if self.main_input_name in input_dict:
            return input_dict[self.main_input_name].numel()
        elif "estimate_tokens" not in self.warnings_issued:
            self.warnings_issued['estimate_tokens'] = True
        return 0
    
    def floating_point_ops(self, input_dict = Dict[str, Union[torch.Tensor, Any]], exclude_embeddings : bool = True) -> int:
        return 6 * self.estimate_tokens(input_dict) * self.num_parameters(exclude_embeddings=exclude_embeddings)


class PreTrainedModel(nn.Module, ModuleUtilsMixin, GenerationMixin, PushToHubMixin, PeftAdapterMixin):
    config_class = None
    base_model_prefix = ""
    main_input_name = "input_ids"
    model_tags = None

    _auto_class = None
    _no_split_modules = None
    _skip_keys_device_placement = None
    _keep_in_fp32_modules = None

    _keys_to_ignore_on_load_missing = None
    _keys_to_ignore_on_load_unexpected = None
    _keys_to_ignore_on_save = None
    _tied_weights_keys = None

    is_parallelizable = False
    supports_gradient_checkpointing = False
    _is_stateful = False

    _supports_flash_attn_2 = False
    _supports_sdpa = False
    _supports_cache_class = False
    _supports_static_cache = False
    _supports_quantized_cache = False

    @property
    def dummy_inputs(self) ->Dict[str, torch.Tensor]:
        return {"input_ids" : torch.tensor(DUMMY_INPUTS)}
    
    @property
    def framework(self) -> str:
        return "pt"
    
    def __init__(self, config : PretrainedConfig, *inputs, **kwargs):
        super().__init__()
        if not isinstance(config, PretrainedConfig):
            raise ValueError(
                f"Parameter config in `{self.__class__.__name__}(config)` should be an instance of class "
                f"`PretrainedConfig`. To create a model from a pretrained model use ")
        
        config = self._autoset_attn_implementation(config, torch_dtype = torch.get_default_dtype(), check_device_map = False)
        self.config = config

        self.name_or_path = config.name_or_path
        self.warnings_issued = {}
        self.generation_config = GenerationConfig.from_model_config(config) if self.can_generate() else None
        self._keep_in_fp32_modules = copy.copy(self.__class__._keep_in_fp32_modules)
    
    def post_init(self):
        self.init_weights()
        self._backward_compatibility_gradient_checkpointing()
    
    def dequantize(self):
        hf_quantizer = getattr(self, "hf_quantizer", None)

        if hf_quantizer is None:
            raise ValueError("Model does not have a quantized weights.")
        
        return hf_quantizer.dequantize(self)
    
    def _backward_compatibility_gradient_checkpointing(self):
        if self.supports_gradient_checkpointing and getattr(self.config, "gradient_checkpointing", False):
            self.gradient_checkpointing_enable()
            delattr(self.config, "gradient_checkpointing")
        
    def add_model_tags(self, tags : Union[List[str], str]) -> None:
        if isinstance(tags, str):
            tags = [tags]
        if self.model_tags is None:
            self.model_tags = []
        
        for tag in tags:
            if tag not in self.model_tags:
                self.model_tags.append(tag)
    
    @classmethod
    def _from_config(cls, config, **kwargs):
        torch_dtype = kwargs.pop("torch_dtype", None)
        use_flash_attention_2 = kwargs.pop("use_flash_attention_2", False)

        dtype_orig = None
        if torch_dtype is not None:
            dtype_orig = cls._set_default_torch_dtype(torch_dtype)
        
        config = copy.deepcopy(config)

        if config._attn_implementation is not None:
            attn_implementation = config._attn_implementation_internal
        else:
            attn_implementation = None
        
        config._attn_implementation = kwargs.pop('attn_implementation', attn_implementation)
        config = cls._autoset_attn_implementation(
            config,
            use_flash_attention_2 = use_flash_attention_2,
            check_device_map = False,
            torch_dtype = torch_dtype
        )

        model = cls(config, **kwargs)
        if dtype_orig is not None:
            torch.set_default_dtype(dtype_orig)
        
        return model
    

    @classmethod
    def _autoset_attn_implementation(cls, config, use_flash_attention_2 : bool = False, torch_dtype : Optional[torch.dtype] = None, device_map : Optional[Union[str, Dict[str, int]]] = None, check_device_map : bool = True):
        requested_attn_implementation = None
        

    