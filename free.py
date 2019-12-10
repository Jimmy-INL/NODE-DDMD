import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from scipy import linalg as la
A = np.array([[1, 2, 3], [3, 4, 6]])
print(A ** 2)


print(torch.sum(torch.tensor([[1, 2], [3, 4]])))

p = [2, 4, 5, 6]
q = []
print(p[3])
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
t = torch.tensor([[1, 2, 6], [3, 4., 2]]); print(t)
# deviceを指定することでGPUにTensorを作成する
# t = torch.tensor([[1, 2], [3, 4.]], device="cuda:0")
# print(t)
# dtypeを指定することで倍精度のTensorを作る
t = torch.tensor([[1, 2], [3, 4.]], dtype=torch.float64); print(t)
# 0から9まで数値で初期化された1次元のTensor
t = torch.arange(0, 10); print(t)
# すべての値が0の100×10のTensorを作成し，toメソッドでGPUに転送する
# t = torch.zeros(100, 10).to("cuda:0")
# 正規乱数で100×10のTensorを作成
t = torch.randn(100, 10); print(t)
# Tensorのshapeはsizeメソッドで習得可能
print(t.size(), "サイズ")

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