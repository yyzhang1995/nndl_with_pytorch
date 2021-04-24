import random
import torch

def data_iter(batch_size, features, labels):
    num_examples = len(labels)
    indices = list(range(num_examples))
    random.shuffle(indices)

    for i in range(0, num_examples, batch_size):
        index_selected = torch.LongTensor(indices[i:min(i + batch_size, num_examples)])
        yield features.index_select(0, index_selected), labels.index_select(0, index_selected)


def squared_loss(y_hat, y):
    return (y_hat - y.view(y_hat.size())) ** 2 / 2


def cross_entropy(y_hat, y):
    return - torch.log(y_hat.gather(1, y.view(-1, 1)))


def sgd(params, lr, batch_size):
    """

    :param params:
    :param lr: 学习率
    :param batch_size:
    :return:
    """
    for param in params:
        param.data -= lr * param.grad / batch_size

def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp / partition


def evaluate_accuracy(data_iter, net, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        if isinstance(net, torch.nn.Module):
            net.eval()
            acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
            net.train()
        else:
            try:
                acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
            except TypeError:
                acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n

def relu(x):
    return torch.max(input=x, other=torch.tensor(0.0))

def corr2d(X, K):
    h, w = K.shape
    Y = torch.zeros(size=(X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i + h, j: j + 2] * K).sum()
    return Y

