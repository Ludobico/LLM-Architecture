import torch
import torch.nn as nn
from packaging import version

ALL_LAYERNORM_LAYERS = [nn.LayerNorm]

parsed_torch_version_base = version.parse(version.parse(torch.__version__).base_version)

is_torch_greater_or_equal_than_2_4 = parsed_torch_version_base >= version.parse("2.4")
is_torch_greater_or_equal_than_2_3 = parsed_torch_version_base >= version.parse("2.3")
is_torch_greater_or_equal_than_2_2 = parsed_torch_version_base >= version.parse("2.2")
is_torch_greater_or_equal_than_2_1 = parsed_torch_version_base >= version.parse("2.1")
is_torch_greater_or_equal_than_2_0 = parsed_torch_version_base >= version.parse("2.0")
is_torch_greater_or_equal_than_1_13 = parsed_torch_version_base >= version.parse("1.13")
is_torch_greater_or_equal_than_1_12 = parsed_torch_version_base >= version.parse("1.12")