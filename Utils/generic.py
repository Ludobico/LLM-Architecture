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
    