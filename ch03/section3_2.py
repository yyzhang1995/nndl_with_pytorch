import torch
import numpy as np
import matplotlib.pyplot as plt
from common.utils import *

# 生成数据集
num_input = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2

features = torch.from_numpy(np.random.normal(0, 1, (num_examples, num_input)))
labels = torch.matmul(features, torch.tensor(true_w, dtype=torch.double)) + true_b
labels += torch.from_numpy(np.random.normal(0, 0.01, size=labels.size()))

# 作图，不需要时注释掉
# plt.scatter(features[:, 1].numpy(), labels.numpy())
# plt.show()

# 编写了逐次获取小批量数据的函数data_iter
batch_size = 10
for X, y in data_iter(batch_size, features, labels):
    print(X, y)
    break

# 初始化模型参数
w = torch.tensor(np.random.normal(0, 0.01, (num_input, 1)))
b = torch.zeros(1, dtype=torch.float32)

w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)


# 自定义模型，线性模型
def linreg(X, w, b):
    return torch.mm(X, w) + b

# 定义损失函数
# 已经定义在common.utils当中

# 定义优化算法
# 随机梯度下降算法已经定义在common.utils当中

# 至此开始训练模型
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y).sum()
        l.backward()
        sgd([w, b], lr, batch_size)

        w.grad.data.zero_()
        b.grad.data.zero_()
    train_l = loss(net(features, w, b), labels)
    print('epoch %d, loss %f' % (epoch + 1, train_l.mean().item()))

print(true_w, '\n', w)
print(true_b, '\n', b)