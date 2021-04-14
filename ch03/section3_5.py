import torchvision
import torchvision.transforms as transforms
from common.utils_fashion_mnist import *
import torch

mnist_train = torchvision.datasets.FashionMNIST(root="../Datasets/FashionMNIST", train=True, download=True,
                                                transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root="../Datasets/FashionMNIST", train=False, download=True,
                                               transform=transforms.ToTensor())

print(mnist_train[0])

X, y = [], []
for i in range(10):
    X.append(mnist_train[i][0])
    y.append(mnist_train[i][1])
show_fashion_mnist(X, get_fashion_mnist_labels(y))

# 批量读取一次数据所花费的时间
import time
import sys

batch_size = 256
if sys.platform.startswith("win"):
    num_workers = 0
else:
    num_workers = 4
train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=True, num_workers=num_workers)

start = time.time()
for X, y in test_iter:
    continue
print('%.2f sec' % (time.time() - start))

# 封装后的测试
train_iter, test_iter = load_data_fashion_mnist(batch_size)
start = time.time()
for X, y in train_iter:
    continue
print('%.2f sec' % (time.time() - start))