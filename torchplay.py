import torch


matrix = torch.randn(4,5)

bias = torch.ones(5)

result = matrix + bias

print("Matrix:\n", matrix)
print("Bias:\n", bias)
print("Result:\n", result)