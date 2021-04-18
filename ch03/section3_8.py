import torch
import numpy as np
import matplotlib.pyplot as plt

def xyplot(x_vals, y_vals, name):
    plt.plot(x_vals.detach().numpy(), y_vals.detach().numpy())
    plt.xlabel("x")
    plt.ylabel(name + "(x)")
    plt.show()

x = torch.arange(-8, 8, 0.1, requires_grad=True)
y = x.relu()
# xyplot(x, y, 'relu')

y.sum().backward()
# xyplot(x, x.grad, 'grad of relu')

x.grad.data.zero_()
y = x.sigmoid()
# xyplot(x, y, 'sigmoid')

y.sum().backward()
# xyplot(x, x.grad, 'grad of sigmoid')

x.grad.data.zero_()
y = x.tanh()
# xyplot(x, y, 'tanh')