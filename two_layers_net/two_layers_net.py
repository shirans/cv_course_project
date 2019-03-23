
# init to zeros
import torch
from torch.autograd import Variable

x = torch.Tensor(2, 3)

# init to random
x = torch.rand(2, 3)


x = Variable(torch.ones(2, 2) * 2, requires_grad=True)
z = 2 * (x * x) + 5 * x


z.backward(torch.ones(2, 2))
print(x.grad)
