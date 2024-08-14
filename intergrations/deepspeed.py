import collections, sys, os
sys.path.append(os.path.dirname(os.path.realpath(os.path.join(__file__, '..'))))

import copy
import importlib.metadata as importlib_metadata
import importlib.util
import weakref
from functools import partialmethod

from Utils.import_utils import is_accelerate_available, is_torch_avaiable, is_torch_mlu_available

if is_torch_avaiable():
    import torch

def is_deepspeed_available():
    package_exists = importlib.util.find_spec("deepspeed") is not None

    if package_exists:
        try:
            if is_torch_mlu_available():
                _ = importlib_metadata.metadata('deepspeed-mlu')
                return True
            _ = importlib_metadata.metadata('deepspeed')
            return True
        except importlib_metadata.PackageNotFoundError:
            return False
    
_hf_deepspeed_config_weak_ref = None

def is_deepspeed_zero3_enabled():
    if _hf_deepspeed_config_weak_ref is not None and _hf_deepspeed_config_weak_ref() is not None:
        return _hf_deepspeed_config_weak_ref().is_zero3()
    else:
        return False