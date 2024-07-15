import torch
import torch.nn as nn

t = torch.randn(3,4)
layer_norm = nn.LayerNorm(t.shape[-1])

print(t)

print(layer_norm(t))