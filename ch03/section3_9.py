import torch
import numpy as np
from common.utils_fashion_mnist import load_data_fashion_mnist
from common.utils import relu, softmax, cross_entropy, sgd, evaluate_accuracy

# 部分超参数
batch_size = 256
num_inputs = 784
num_outputs = 10
num_hidden = 256

# 读取数据
train_iter, test_iter = load_data_fashion_mnist(256)

# 初始化参数
W1 = torch.tensor(np.random.normal(size=(num_inputs, num_hidden), loc=0.0, scale=0.01),
                  dtype=torch.float, requires_grad=True)
b1 = torch.zeros(num_hidden, dtype=torch.float, requires_grad=True)
W2 = torch.tensor(np.random.normal(size=(num_hidden, num_outputs), loc=0.0, scale=0.01),
                  dtype=torch.float, requires_grad=True)
b2 = torch.zeros(num_outputs, dtype=torch.float, requires_grad=True)
params = [W1, b1, W2, b2]

# 定义网络结构
def net(X):
    u1 = torch.matmul(X.view(X.shape[0], -1), W1) + b1
    a1 = relu(u1)
    return torch.matmul(a1, W2) + b2

# 定义损失函数
loss = torch.nn.CrossEntropyLoss()

# 定义优化器
# 已经定义了优化器sgd

# 训练网络
epochs = 5
lr = 100
for epoch in range(epochs):
    train_loss_sum, train_acc_num, n = 0.0, 0.0, 0
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        for param in params:
            if param.grad is not None:
                param.grad.data.zero_()
        l.backward()
        sgd(params, lr, batch_size)
        train_loss_sum += l.item()
        train_acc_num += (y_hat.argmax(dim=1) == y).sum().item()
        n += y.shape[0]
    test_accuracy = evaluate_accuracy(test_iter, net)
    print("epoch: %d, train loss: %.4f, train accuracy rate: %.3f, test accuracy: %.3f" % \
          (epoch + 1, train_loss_sum / n, train_acc_num / n, test_accuracy))
