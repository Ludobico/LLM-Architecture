import inspect
import tempfile
import warnings
from collections import OrderedDict, UserDict
from collections.abc import MutableMapping
from contextlib import ExitStack, contextmanager
from dataclasses import fields, is_dataclass
from enum import Enum
from functools import partial, wraps
from typing import Any, ContextManager, Iterable, List, Optional, Tuple

import numpy as np
from packaging import version

from .import_utils import (
    get_torch_version,
    is_flex_available,
    is_mlx_available,
    is_tf_available,
    is_torch_avaiable,
    is_torch_fx_proxy,   
)

class ExplicitEnum(str, Enum):
    @classmethod
    def _missing_(cls, value) -> Any:
        raise ValueError(f"{value} is not a valud {cls.__name__}, Please select one of {list(cls._value2member_map_.keys())}")

class PaddingStrategy(ExplicitEnum):
    LONGEST = "longest"
    MAX_LENGTH = "max_length"
    DO_NOT_PAD = "do_not_pad"


def add_model_info_to_auto_map(auto_map, repo_id):
    for key, value in auto_map.items():
        if isinstance(value, (tuple, list)):
            auto_map[key] = [f"{repo_id}--{v}" if (v is not None and "--" not in v) else v for v in value]
        elif value is not None and '--' not in value:
            auto_map[key] = f"{repo_id}--{value}"

def add_model_info_to_custom_pipelines(custom_pipeline, repo_id):
    for task in custom_pipeline.keys():
        if "impl" in custom_pipeline[task]:
            module = custom_pipeline[task]["impl"]
            if "--" not in module:
                custom_pipeline[task]["impl"] = f"{repo_id}--{module}"
    return custom_pipeline

@contextmanager
def working_or_temp_dir(working_dir, use_temp_dir : bool = False):
    if use_temp_dir:
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield tmp_dir
    else:
        yield working_dir