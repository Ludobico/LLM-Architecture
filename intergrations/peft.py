import sys, os
sys.path.append(os.path.dirname(os.path.realpath(os.path.join(__file__, '..'))))
import inspect
import warnings

from typing import Any, Dict, List, Optional, Union

from Utils.import_utils import is_torch_available, is_accelerate_available, is_peft_available
from intergrations.peft_utils import find_adapter_config_file, check_peft_version

from peft import PeftConfig, inject_adapter_in_model, load_peft_weights
from peft.utils import set_peft_model_state_dict


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
    
        inject_adapter_in_model(peft_config, self, adapter_name)

        if not self._hf_peft_config_loaded:
            self._hf_peft_config_loaded = True
        
        if peft_model_id is not None:
            adapter_state_dict = load_peft_weights(peft_model_id, token=token, device=device, **adapter_kwargs)

        processed_adapter_state_dict = {}
        prefix = "base_mode.model."
        for key, value in adapter_state_dict.items():
            if key.startswith(prefix):
                new_key = key[len(prefix) :]
            else:
                new_key = key
            
            processed_adapter_state_dict[new_key] = value

        incompatible_keys = set_peft_model_state_dict(self, processed_adapter_state_dict, adapter_name)

        if incompatible_keys is not None:
            if hasattr(incompatible_keys, 'unexpected_keys') and len(incompatible_keys.unexpected_keys) > 0:
                warnings(warninig="The following keys in the state_dict are not present in the model: " + str(incompatible_keys.unexpected_keys))
        
        if (
            (getattr(self, 'hf_device_map', None) is not None)
            and (len(set(self._hf_device_map.values()).intersection({'cpu', 'disk'})) > 0)
            and len(self.peft_config) == 1
        ):
            self._dispatch_accelerate_model(
                device_map=device_map,
                max_memory=max_memory,
                offload_folder=offload_folder,
                offload_index=offload_index,
            )
    
    def add_adapter(self, adapter_config, adapter_name : Optional[str] = None) -> None:
        check_peft_version(min_version=MIN_PEFT_VERSION)

        from peft import PeftConfig, inject_adapter_in_model

        adapter_name = adapter_name or 'default'

        if not self._hf_peft_config_loaded:
            self._hf_peft_config_loaded = True
        elif adapter_name in self.peft_config:
            raise ValueError(f"Adapter {adapter_name} already loaded")
        
        if not isinstance(adapter_config, PeftConfig):
            raise TypeError(f"Adapter config must be of type PeftConfig, got {type(adapter_config)}")
        
        adapter_config.base_model_name_or_path = self.__dict__.get("name_or_path", None)
        inject_adapter_in_model(adapter_config, self, adapter_name)

        self.set_adapter(adapter_name)
    
    def set_adapter(self, adapter_name : Union[List[str], str]) -> None:
        check_peft_version(min_version=MIN_PEFT_VERSION)

        if not self._hf_peft_config_loaded:
            raise ValueError("Please call `add_adapter` before `set_adapter`")
        elif isinstance(adapter_name, list):
            missing = set(adapter_name) - set(self.peft_config)
            if len(missing) > 0:
                raise ValueError(f"Missing adapters: {missing}")
        elif adapter_name not in self.peft_config:
            raise ValueError(f"Adapter {adapter_name} not loaded")
        
        from peft.tuners.tuners_utils import BaseTunerLayer
        from peft.utils import ModulesToSaveWrapper

        _adapters_has_been_set = False

        for _, module in self.named_modules():
            if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper)):
                if hasattr(module, 'set_adapter'):
                    module.set_adapter(adapter_name)
                else:
                    module.active_adapter = adapter_name
                _adapters_has_been_set = True

        if not _adapters_has_been_set:
            raise ValueError("No module in the model has an adapter set")
        
    def disable_adapters(self) -> None:
        check_peft_version(MIN_PEFT_VERSION)

        if not self._hf_peft_config_loaded:
            raise ValueError("Please call `add_adapter` before `disable_adapters`")
        
        from peft.tuners.tuners_utils import BaseTunerLayer
        from peft.utils import ModulesToSaveWrapper

        for _, module in self.named_modules():
            if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper)):
                # The recent version of PEFT need to call `enable_adapters` instead
                if hasattr(module, "enable_adapters"):
                    module.enable_adapters(enabled=False)
                else:
                    module.disable_adapters = True

    def enable_adapters(self) -> None:
        """
        If you are not familiar with adapters and PEFT methods, we invite you to read more about them on the PEFT
        official documentation: https://huggingface.co/docs/peft

        Enable adapters that are attached to the model. The model will use `self.active_adapter()`
        """
        check_peft_version(min_version=MIN_PEFT_VERSION)

        if not self._hf_peft_config_loaded:
            raise ValueError("No adapter loaded. Please load an adapter first.")

        from peft.tuners.tuners_utils import BaseTunerLayer

        for _, module in self.named_modules():
            if isinstance(module, BaseTunerLayer):
                # The recent version of PEFT need to call `enable_adapters` instead
                if hasattr(module, "enable_adapters"):
                    module.enable_adapters(enabled=True)
                else:
                    module.disable_adapters = False

