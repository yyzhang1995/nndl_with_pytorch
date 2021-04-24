import torch
import torch.nn as nn
from common.utils import corr2d

def corr2d_multi_in(X, K):
    res = corr2d(X[0, :, :], K[0, :, :])
    for i in range(1, X.shape[0]):
        res += corr2d(X[i, :, :], K[i, :, :])
    return res

X = torch.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                 [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
K = torch.tensor([[[0, 1], [2, 3]],
                  [[1, 2], [3, 4]]])

# print(corr2d_multi_in(X, K))

def corr2d_multi_in_out(X, K):
    return torch.stack([corr2d_multi_in(X, k) for k in K])

K = torch.stack([K, K + 1, K + 2])
# print(K.shape)
# for k in K:
#     print(k.shape)
print(corr2d_multi_in_out(X, K))

# 1 * 1的卷积核
def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_out = K.shape[0]
    Y = torch.matmul(K.view(c_out, c_i), X.view(c_i, h * w))
    return Y.view(c_out, h, w)

X = torch.rand(3, 3, 3)
K = torch.rand(2, 3, 1, 1)

Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out_1x1(X, K)
print(Y1 - Y2)