from torch.autograd import Variable
import torch

x_tensor=torch.randn(10,5)
x=Variable(x_tensor,requires_grad=True)
print(x.data)
print(x.grad)
print(x.grad_fn)