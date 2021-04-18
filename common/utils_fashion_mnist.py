import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import torch
import sys


def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def show_fashion_mnist(images, labels):
    _, figs = plt.subplots(1, len(images), figsize=(12,12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()


def load_data_fashion_mnist(batch_size, num_workers=4):
    if sys.platform.startswith("win"):
        num_workers = 0
    root = r'E:\workspace\pycharm\Pyproject03_nndl_pytorch\Datasets\FashionMNIST'
    mnist_train = torchvision.datasets.FashionMNIST(root=root,
                                                    train=True, download=False,
                                                    transform=transforms.ToTensor())
    mnist_test = torchvision.datasets.FashionMNIST(root=root, train=False, download=False,
                                                   transform=transforms.ToTensor())
    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return train_iter, test_iter
