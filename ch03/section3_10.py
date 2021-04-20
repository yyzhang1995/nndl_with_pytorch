import torch
from common.utils_fashion_mnist import load_data_fashion_mnist
import torch.nn as nn
from common.utils_layers import FlattenLayer
from common.utils import evaluate_accuracy

batch_size = 256
num_inputs, num_hidden, num_outputs = 784, 256, 10

train_iter, test_iter = load_data_fashion_mnist(batch_size)

# 定义网络结构
net = nn.Sequential(
    FlattenLayer(),
    nn.Linear(in_features=num_inputs, out_features=num_hidden),
    nn.ReLU(),
    nn.Linear(in_features=num_hidden, out_features=num_outputs)
)

# 定义损失函数
loss = nn.CrossEntropyLoss()

# 定义优化器
import torch.optim as optim
optimizer = optim.SGD(net.parameters(), lr=0.5)

# 训练模型
epochs = 5
for epoch in range(epochs):
    train_loss_sum = 0.0
    train_acc = 0.0
    n = 0
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)

        optimizer.zero_grad()

        l.backward()
        optimizer.step()

        train_loss_sum += l.item()
        train_acc += (y_hat.argmax(dim=1) == y).sum().item()
        n += y.shape[0]

    test_accuracy = evaluate_accuracy(test_iter, net)
    print("epoch: %d, train loss: %.3f, train acc: %.3f, test acc: %.3f" % \
          (epoch + 1, train_loss_sum / n, train_acc / n, test_accuracy))
