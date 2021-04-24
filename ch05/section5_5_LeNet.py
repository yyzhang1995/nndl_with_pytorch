import torch
import torch.optim as optim
import torch.nn as nn
from common.utils_fashion_mnist import load_data_fashion_mnist
from common.utils import evaluate_accuracy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output


# 定义训练参数
batch_size = 256
lr = 0.001

# 定义网络结构
net = LeNet()
print(net)

# 获取数据
train_iter, test_iter = load_data_fashion_mnist(batch_size)

# 定义优化器
optimizer = optim.Adam(net.parameters(), lr=lr)

# 定义损失函数
loss = torch.nn.CrossEntropyLoss()

# 训练网络
net = net.to(device)
num_epochs = 5
for epoch in range(num_epochs):
    train_loss_sum, train_acc_sum, n = 0.0, 0.0, 0
    for X, y in train_iter:
        X = X.to(device)
        y = y.to(device)
        y_hat = net(X)
        l = loss(y_hat, y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()

        train_loss_sum += l.cpu().item()
        train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
        n += y.shape[0]
    test_acc = evaluate_accuracy(test_iter, net)
    print("epoch: %d, loss: %.4f, train acc %.3f, test acc %.3f" %
          (epoch + 1, train_loss_sum / n, train_acc_sum / n, test_acc))
