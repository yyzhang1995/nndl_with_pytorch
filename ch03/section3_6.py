import torch
import torchvision
import numpy as np
from common.utils_fashion_mnist import *
from common.utils import softmax, cross_entropy, evaluate_accuracy, sgd

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
# print(evaluate_accuracy(train_iter, net))

epoches = 5
lr = 0.1
params = [W, b]
for epoch in range(epoches):
    train_loss_sum = 0.0
    train_acc_sum = 0.0
    n = 0
    for X, y in train_iter:
        y_hat = net(X)
        loss = cross_entropy(y_hat, y).sum()
        # 梯度清零
        for param in params:
            if param.grad is not None: param.grad.data.zero_()
        # 反向传播
        loss.backward()
        sgd(params, lr, batch_size)
        train_loss_sum += loss.item()
        train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
        n += y.shape[0]
    test_accuracy = evaluate_accuracy(test_iter, net)
    print("epoch %d, loss %.4f, train acc %.3f, test acc %.3f" % \
          (epoch + 1, train_loss_sum / n, train_acc_sum / n, test_accuracy))


# 作业练习，请忽略
# X = torch.tensor([[[1, 2], [3, 4]]], dtype=torch.float)
# X.requires_grad_(requires_grad=True)
# u = torch.nn.MaxPool2d(kernel_size=2)(X)
# y = u.sum()
# y.backward()
# print(X.grad)