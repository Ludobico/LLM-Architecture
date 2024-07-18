import torch
import torch.nn as nn


t1 = torch.randn(2,3,1,2,3)
t2 = torch.randn(2,25,1,4,3)

tc = torch.cat((t1, t2), dim=1)

print(tc.shape)