import torch.nn as nn
from common.utils_fashion_mnist import load_data_fashion_mnist
from common.utils import evaluate_accuracy

# 定义部分超参数
batch_size = 256
num_inputs = 784
num_outputs = 10

# 导入数据
train_iter, test_iter = load_data_fashion_mnist(batch_size)

# 定义网络结构
class SoftmaxNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(SoftmaxNet, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)

    def forward(self, x):
        y = self.linear(x.view(x.shape[0], -1))
        return y

net = SoftmaxNet(num_inputs, num_outputs)

# 初始化模型参数
import torch.nn.init as init
init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0.0)

# 定义损失函数
loss = nn.CrossEntropyLoss()

# 定义优化器
import torch.optim as optim
optimizer = optim.SGD(net.parameters(), lr=0.1)

# 训练模型
epoches = 5
for epoch in range(epoches):
    train_loss_sum = 0.0
    train_acc_num = 0.0
    n = 0
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)

        optimizer.zero_grad()
        l.backward()
        optimizer.step()

        train_loss_sum += l.item()
        train_acc_num += (y_hat.argmax(dim=1) == y).sum().item()
        n += y.shape[0]
    test_accuracy = evaluate_accuracy(test_iter, net)
    print("epoch: %d, train loss: %.4f, train accuracy rate: %.3f, test accuracy: %.3f" %\
          (epoch + 1, train_loss_sum / n, train_acc_num / n, test_accuracy))
