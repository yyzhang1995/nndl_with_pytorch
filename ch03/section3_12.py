import torch
import numpy as np
from common.utils import squared_loss, sgd
import matplotlib.pyplot as plt

batch_size = 1
n_train, n_test, num_inputs = 20, 100, 200
true_w, true_b = torch.ones(num_inputs, dtype=torch.float) * 0.01, torch.tensor(0.05)

# 生成数据
features = torch.randn(size=(n_train + n_test, num_inputs))
labels = torch.matmul(features, true_w) + true_b
labels += torch.tensor(np.random.normal(size=labels.size(), loc=0, scale=0.01), dtype=torch.float)
train_features, train_labels = features[:n_train, :], labels[: n_train]
test_features, test_labels= features[n_train:, :], labels[n_train:]

data = torch.utils.data.TensorDataset(train_features, train_labels)
train_iter = torch.utils.data.DataLoader(data, batch_size, shuffle=True)

# 初始化参数
def init_params():
    w = torch.tensor(np.random.normal(size=(num_inputs, 1)), dtype=torch.float, requires_grad=True)
    b = torch.zeros(1, requires_grad=True, dtype=torch.float)
    return [w, b]

# 定义正则化项
def l2_penalty(w):
    return (w * w).sum() / 2

# 定义损失函数
loss = squared_loss

# 定义网络
def net(X, w, b):
    return torch.matmul(X, w) + b

# 训练模型
def fit_and_plot(lambd):
    epochs = 100
    lr = 0.003
    params = init_params()
    train_ls, test_ls = [], []
    for epoch in range(epochs):
        for X, y in train_iter:
            l = loss(net(X, *params), y) + lambd * l2_penalty(params[0])
            l = l.sum()
            for param in params:
                if param.grad is not None:
                    param.grad.data.zero_()
            l.backward()
            sgd(params, lr, batch_size)
        train_ls.append(loss(net(train_features, *params), train_labels).mean().item())
        test_ls.append(loss(net(test_features, *params), test_labels).mean().item())
    plt.semilogy(range(1, epochs + 1), train_ls, label='train loss')
    plt.semilogy(range(1, epochs + 1), test_ls, label='test loss')
    plt.legend()
    plt.show()
    print('L2 norm of w:', params[0].norm().item())

def complex_main():
    # fit_and_plot(0)
    fit_and_plot(3)

# 一下是简洁实现的方法
def fit_and_plot_pytorch(wd):
    epochs = 100
    lr = 0.003
    net = torch.nn.Linear(num_inputs, 1)
    torch.nn.init.normal_(net.weight, 0, 0.01)
    torch.nn.init.constant_(net.bias, val=0)
    optimizer_w = torch.optim.SGD(params=[net.weight], lr=lr, weight_decay=wd)
    optimizer_b = torch.optim.SGD(params=[net.bias], lr=lr)

    train_ls, test_ls = [], []
    for epoch in range(epochs):
        for X, y in train_iter:
            l = loss(net(X), y).mean()
            optimizer_w.zero_grad()
            optimizer_b.zero_grad()
            l.backward()
            optimizer_w.step()
            optimizer_b.step()
        train_ls.append(loss(net(train_features), train_labels).mean().item())
        test_ls.append(loss(net(test_features), test_labels).mean().item())
    plt.semilogy(range(1, epochs + 1), train_ls, label='train loss')
    plt.semilogy(range(1, epochs + 1), test_ls, label='test loss')
    plt.legend()
    plt.show()
    print(net.weight.norm().item())


def simple_main():
    fit_and_plot_pytorch(wd=3)

if __name__ == '__main__':
    simple_main()