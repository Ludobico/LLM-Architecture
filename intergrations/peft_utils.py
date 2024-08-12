import importlib.metadata
import sys, os
sys.path.append(os.path.dirname(os.path.realpath(os.path.join(__file__, '..'))))

import importlib
from typing import Dict, Optional, Union

from packaging import version
from Utils.import_utils import is_peft_available

ADAPTER_CONFIG_NAME = "adapter_config.json"
ADAPTER_WEIGHTS_NAME = "adapter_model.bin"
ADAPTER_SAFE_WEIGHTS_NAME = "adapter_model.safetensors"

def find_adapter_config_file(
        model_id : str,
        cache_dir : Optional[Union[str, os.PathLike]] = None,
        force_download : bool = False,
        resume_download : Optional[bool] = None,
        proxies : Optional[Dict[str, str]] = None,
        token : Optional[Union[bool, str]] = None,
        revision : Optional[str] = None,
        local_files_only : bool = False,
        subfolder : str = "",
        _commit_hash : Optional[str] = None
) -> Optional[str]:
    adapter_cached_filename = None
    if model_id is None:
        return None
    
    elif os.path.isdir(model_id):
        list_remote_files = os.listdir(model_id)
        if ADAPTER_CONFIG_NAME in list_remote_files:
            adapter_cached_filename = os.path.join(model_id, ADAPTER_CONFIG_NAME)
    
    return adapter_cached_filename

def check_peft_version(min_version : str) -> None:
    if not is_peft_available():
        raise ValueError(
            f"PEFT is not installed. Please install it with `pip install peft -U`")
    
    is_peft_version_compatible = version.parse(importlib.metadata.version("peft")) >= version.parse(min_version)

    if not is_peft_version_compatible:
        raise ValueError(
            f"PEFT version {importlib.metadata.version('peft')} is not compatible with the required version {min_version}. Please upgrade it with `pip install peft -U`"
        )