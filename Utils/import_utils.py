import importlib.metadata
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import warnings
from collections import OrderedDict
from functools import lru_cache
from itertools import chain
from types import ModuleType
from typing import Any, Tuple, Union

from packaging import version

def _is_package_available(pkg_name : str, return_version : bool = False) -> Union[Tuple[bool, str], bool]:
    package_exists = importlib.util.find_spec(pkg_name) is not None
    package_version = "N/A"
    if package_exists:
        try:
            package_version = importlib.metadata.version(pkg_name)
        except importlib.metadata.PackageNotFoundError:
            if pkg_name == "torch":
                try:
                    package = importlib.import_module(pkg_name)
                    temp_version = getattr(package, "__version__", "N/A")

                    if "dev" in temp_version:
                        package_version = temp_version
                        package_exists = True
                    else:
                        package_exists = False
                
                except ImportError:
                    package_exists = False
    
    else:
        return package_exists
    

ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}
EVN_VARS_TRUE_AND_AUTO_VALUES = ENV_VARS_TRUE_VALUES.union({"AUTO"})

USE_TF = os.environ.get("USE_TF", "AUTO").upper()
USE_TORCH = os.environ.get("USE_TORCH", "AUTO").upper()
USE_JAX = os.environ.get("USE_FLAX", "AUTO").upper()

USE_TORCH_XLA = os.environ.get("USE_TORCH_XLA", "1").upper()

FORCE_TF_AVAILABLE = os.environ.get("FORCE_TF_AAVAILABLE", "AUTO").upper()

TORCH_FX_REQUIRED_VERSION = version.parse("1.10")

ACCELERATE_MIN_VERSION = "0.21.0"
FSDP_MIN_VERSION = "1.12.0"
XLS_FSDPV2_MIN_VERSION = "2.2.0"


_accelerate_available, _accelerate_version = _is_package_available("accelerate", return_version=True)
_apex_available = _is_package_available("apex")
_aqlm_available = _is_package_available("aqlm")
_av_available = importlib.util.find_spec("av") is not None
_bitsandbytes_available = _is_package_available("bitsandbytes")
_eetq_available = _is_package_available("eetq")
_galore_torch_available = _is_package_available("galore_torch")
_lomo_available = _is_package_available("lomo")
_bs4_available = importlib.util.find_spec("bs4") is not None
_coloredlogs_available = _is_package_available("coloredlogs")
_cv2_available = importlib.util.find_spec("cv2")
_datasets_available = _is_package_available("datasets")
_decord_avaulable = importlib.util.find_spec("decord") is not None
_detectron2_available = _is_package_available("detectron2")
_faisee_available = importlib.util.find_spec("faisee") is not None
_mlx_available = _is_package_available("mlx")

_torch_version = "N/A"
_torch_available = False
if USE_TORCH in EVN_VARS_TRUE_AND_AUTO_VALUES and USE_TF not in ENV_VARS_TRUE_VALUES:
    _torch_available, _torch_version = _is_package_available("torch", return_version=True)

_flax_available = False
if USE_JAX in EVN_VARS_TRUE_AND_AUTO_VALUES:
    _flax_available, _flax_version = _is_package_available("flax", return_version=True)
    if _flax_available:
        _jax_available, _jax_version = _is_package_available("jax", return_version=True)

_tf_version = "N/A"
_tf_available = False

if FORCE_TF_AVAILABLE in ENV_VARS_TRUE_VALUES:
    _tf_available = True

_torch_fx_available = False
if _torch_available:
    torch_version = version.parse(_torch_version)
    _torch_fx_available = (torch_version.major, torch_version.minor) >= (
        TORCH_FX_REQUIRED_VERSION.major,
        TORCH_FX_REQUIRED_VERSION.minor
    )

def get_torch_version():
    return _torch_version

def is_flex_available():
    return _flax_available

def is_mlx_available():
    return _mlx_available

def is_tf_available():
    return _tf_available

def is_torch_avaiable():
    return _torch_available

def is_torch_fx_available():
    return _torch_fx_available

def is_torch_fx_proxy(x):
    if is_torch_fx_available():
        import torch.fx

        return isinstance(x, torch.fx.Proxy)
    return False