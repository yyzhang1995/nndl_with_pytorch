import torch
from common.utils_fashion_mnist import load_data_fashion_mnist
from common.utils import evaluate_accuracy
from common.utils_layers import FlattenLayer

batch_size = 256
num_inputs, num_hidden1, num_hidden2, num_outputs = 784, 256, 256, 10
drop_prob1, drop_prob2 = 0.2, 0.5

train_iter, test_iter = load_data_fashion_mnist(batch_size)

# 定义网络结构
net = torch.nn.Sequential(
    FlattenLayer(),
    torch.nn.Linear(num_inputs, num_hidden1),
    torch.nn.ReLU(),
    torch.nn.Dropout(drop_prob1),
    torch.nn.Linear(num_hidden1, num_hidden2),
    torch.nn.ReLU(),
    torch.nn.Dropout(drop_prob2),
    torch.nn.Linear(num_hidden2, num_outputs)
)

# 初始化参数
for param in net.parameters():
    torch.nn.init.normal_(param, mean=0, std=0.01)

# 定义优化器
optimizer = torch.optim.SGD(net.parameters(), lr=0.5)

# 定义损失函数
loss = torch.nn.CrossEntropyLoss()

# 训练模型
epochs = 5
for epoch in range(epochs):
    train_loss_sum, train_acc_sum, n = 0.0, 0.0, 0
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        train_loss_sum += l.item()
        train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
        n += y.shape[0]
    test_acc = evaluate_accuracy(test_iter, net)
    print("epoch: %d, train loss: %.4f, train acc: %.3f, test acc: %.3f" %
          (epoch + 1, train_loss_sum / n, train_acc_sum / n, test_acc))
