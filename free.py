import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from scipy import linalg as la

"""x = torch.randn(2,2)
y = x + 2
print(y.grad_fn)#None / xのrequires_gradがFalseである(演算が追跡されていない)ため

x = torch.randn(2,2,requires_grad=True)
print(x.grad_fn)
y = x + 2
print(y.grad_fn)#<AddBackward0 object at 0x101b40e48>
z = x/2
print(z.grad_fn)#<DivBackward0 object at 0x10432beb8>
"""

import torch.optim as optim
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)


net = Net()
print(net)
input = torch.randn(1, 1, 32, 32)
output = net(input)
target = torch.randn(10)  # a dummy target, for example
target = target.view(1, -1)  # make it the same shape as output
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)
# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

# in your training loop:
optimizer.zero_grad()   # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # Does the update