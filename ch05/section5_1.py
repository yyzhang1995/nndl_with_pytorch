import torch
from common.utils import corr2d
import torch.nn as nn

X = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
K = torch.tensor([[0, 1], [2, 3]])
print(corr2d(X, K))

# 自己实现的2维卷积层
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super(Conv2D, self).__init__()
        self.weight = nn.Parameter(torch.randn(kernel_size))
        self.bias = nn.Parameter(torch.randn(1))

    def forward(self, X):
        return corr2d(X, self.weight) + self.bias

X = torch.ones((6, 8))
X[:, 2:6] = 0

K = torch.tensor([[1, -1]])
Y = corr2d(X, K)
print(Y)

conv2d = Conv2D((1, 2))
lr = 0.01
step = 20

for i in range(step):
    Y_hat = conv2d(X)
    l = ((Y - Y_hat) ** 2).sum()
    l.backward()

    conv2d.weight.data -= lr * conv2d.weight.grad
    conv2d.bias.data -= lr * conv2d.bias.grad

    conv2d.weight.grad.data.zero_()
    conv2d.bias.grad.data.zero_()
    if (i + 1) % 5 == 0:
        print("Step %d, loss %.3f" % (i + 1, l.item()))

print("weight", conv2d.weight.data)
print("bias", conv2d.bias.data)