import torch


x = torch.tensor([[1,2,3], [4,5,6], [7,8,9]])
print(x)
print('-'*80)
print(torch.triu(x))
print('-'*80)
print(torch.triu(x, diagonal=1))
print('-'*80)
print(torch.triu(x, diagonal=-1))