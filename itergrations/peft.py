import sys, os
sys.path.append(os.path.dirname(os.path.realpath(os.path.join(__file__, '..'))))
import inspect
import warnings

from typing import Any, Dict, List, Optional, Union

from Utils.import_utils import is_torch_available, is_accelerate_available, is_peft_available
from itergrations.peft_utils import find_adapter_config_file, check_peft_version


if is_torch_available():
    import torch

if is_accelerate_available():
    from accelerate import dispatch_model
    from accelerate.utils import get_balanced_memory, infer_auto_device_map

MIN_PEFT_VERSION = "0.5.0"

class PeftAdapterMixin:

    _hf_peft_config_loaded = False

    def load_adapter(
            self,
            peft_model_id : Optional[str] = None,
            adapter_name : Optional[str] = None,
            revision : Optional[str] = None,
            token : Optional[str] = None,
            device_map : Optional[str] = 'auto',
            max_memory : Optional[str] = None,
            offload_folder : Optional[str] = None,
            offload_index : Optional[int] = None,
            peft_config : Dict[str, Any ] = None,
            adapter_state_dict : Optional[Dict[str, "torch.Tensor"]] = None,
            adapter_kwargs : Optional[Dict[str, Any]] = None,
    ) -> None:
        check_peft_version(min_version=MIN_PEFT_VERSION)

        adapter_name = adapter_name if adapter_name is not None else "default"
        if adapter_kwargs is None:
            adapter_kwargs = {}
        
        from peft import PeftConfig, inject_adapter_in_model, load_peft_weights
        from peft.utils import set_peft_model_state_dict

        if self._hf_peft_config_loaded and adapter_name in self.peft_config:
            raise ValueError(f"Adapter {adapter_name} already loaded")
        
        if peft_model_id is None and (adapter_state_dict is None and peft_config is None):
            raise ValueError("You need to specify either `peft_model_id`, `peft_config` or `adapter_state_dict`")
        
        if "device" not in adapter_kwargs:
            device = self.device if not hasattr(self, 'hf_device_map') else list(self.hf_device_map.values())[0]
        else:
            device = adapter_kwargs.pop('device')
        
        if isinstance(device, torch.device):
            device = str(device)

        if revision is not None and "revision" not in adapter_kwargs:
            adapter_kwargs['revision'] = revision
        elif revision is not None and "revision" in adapter_kwargs and revision != adapter_kwargs['revision']:
            warnings.warn(f"Overriding revision {adapter_kwargs['revision']} with {revision}")
        
        if "token" in adapter_kwargs:
            token = adapter_kwargs.pop("token")
        
        if peft_config is None:
            adapter_config_file = find_adapter_config_file(peft_model_id, token=token, **adapter_kwargs)

            if adapter_config_file is None:
                raise ValueError(f"Could not find adapter config file for {peft_model_id}")
            
            peft_config = PeftConfig.from_pretrained(adapter_config_file, token=token, **adapter_kwargs)
