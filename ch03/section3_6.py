import torch
import torchvision
import numpy as np
from common.utils_fashion_mnist import *
from common.utils import softmax, cross_entropy, evaluate_accuracy

# 读取数据集
batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)

num_inputs = 784
num_outputs = 10

W = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)), dtype=torch.float)
b = torch.zeros(num_outputs, dtype=torch.float)

W.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)


# 定义前向传播，即模型
def net(X):
    return softmax(torch.mm(X.view((-1, num_inputs)), W) + b)

# 定义损失函数，已经定义在了common.utils当中

# 计算分类准确率
# def accuracy(y_hat, y):
#     return (y_hat.argmax(dim=1) == y).float().mean().item()
# y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
# y = torch.tensor([0, 2])
# print(accuracy(y_hat, y))

print(evaluate_accuracy(train_iter, net))