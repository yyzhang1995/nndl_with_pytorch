import numpy as np
import torch
import torch.utils.data as Data
import torch.nn as nn

num_input = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.tensor(np.random.normal(0, 1, size=(num_examples, num_input)), dtype=torch.float)
labels = torch.matmul(features, torch.tensor(true_w, dtype=torch.float)) + true_b
labels += torch.from_numpy(np.random.normal(0, 0.01, size=labels.size()))

# 读取与处理数据，使用torch自带的包
batch_size = 10
dataset = Data.TensorDataset(features, labels)
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)

for X, y in data_iter:
    print(X, y)
    break

# 定义模型

class LinearNet(nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_feature, 1)

    def foward(self, x):
        y = self.linear(x)
        return y

net = LinearNet(num_input)
print(net)

# 其他写法
net = nn.Sequential(nn.Linear(num_input, 1))

# 写法2
net = nn.Sequential()
net.add_module('linear', nn.Linear(num_input, 1))

for param in net.parameters():
    print(param)

# 初始化模型参数
from torch.nn import init
init.normal_(net[0].weight, mean=0, std=0.01)
init.constant_(net[0].bias, val=0)

# 定义损失函数
loss = nn.MSELoss()

# 定义优化算法
import torch.optim as optim
optimizer = optim.SGD(net.parameters(), lr=0.03)
print(optimizer)

# 训练模型
num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        output = net(X)
        l = loss(output, y.view(-1, 1))
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    print("epoch %d, loss : %f " % (epoch, l.item()))