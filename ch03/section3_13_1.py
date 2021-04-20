import torch
import numpy as np
from common.utils import sgd, evaluate_accuracy
from common.utils_fashion_mnist import load_data_fashion_mnist

def dropout(X, drop_prob):
    X = X.float()
    assert 0 <= drop_prob <= 1
    keep_prob = 1 - drop_prob
    m = (torch.rand(size=X.size()) < keep_prob).float()
    return m * X / keep_prob

# 定义模型超参数
num_inputs, num_hidden1, num_hidden2, num_outputs = 784, 256, 256, 10
batch_size = 256

train_iter, test_iter = load_data_fashion_mnist(batch_size=batch_size)

# 初始化模型参数
w1 = torch.tensor(np.random.normal(size=(num_inputs, num_hidden1), loc=0.0, scale=0.01), dtype=torch.float,
                  requires_grad=True)
b1 = torch.zeros(num_hidden1, dtype=torch.float, requires_grad=True)
w2 = torch.tensor(np.random.normal(size=(num_hidden1, num_hidden2), loc=0.0, scale=0.01), dtype=torch.float,
                  requires_grad=True)
b2 = torch.zeros(num_hidden2, dtype=torch.float, requires_grad=True)
w3 = torch.tensor(np.random.normal(size=(num_hidden2, num_outputs), loc=0.0, scale=0.01), dtype=torch.float,
                  requires_grad=True)
b3 = torch.zeros(num_outputs, dtype=torch.float, requires_grad=True)

params = [w1, b1, w2, b2, w3, b3]

# 定义网络结构
def net(X, is_training=True):
    X = X.view(-1, num_inputs)
    H1 = (torch.matmul(X, w1) + b1).relu()
    if is_training:
        H1 = dropout(H1, 0.2)
    H2 = (torch.matmul(H1, w2) + b2).relu()
    if is_training:
        H2 = dropout(H2, 0.5)
    return torch.matmul(H2, w3) + b3

# 定义损失函数
loss = torch.nn.CrossEntropyLoss()

# 训练模型
lr = 100
num_epochs = 5

for epoch in range(num_epochs):
    train_loss_sum, train_acc_sum, n = 0.0, 0.0, 0
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        for param in params:
            if param.grad is not None:
                param.grad.data.zero_()
        l.backward()
        sgd(params, batch_size=batch_size, lr=lr)
        train_loss_sum += l
        train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
        n += y.shape[0]
    test_acc = evaluate_accuracy(test_iter, net)
    print("epoch: %d, train loss: %.4f, train acc: %.3f, test acc: %.3f" % \
          (epoch + 1, train_loss_sum / n, train_acc_sum / n, test_acc))
