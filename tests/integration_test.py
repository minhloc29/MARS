import torch

x = torch.zeros(5, 3, 1) # only dim with size = 1 can be expanded.
x = x.expand(..., 4)

print(x)