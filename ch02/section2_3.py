import torch

# 自动计算梯度
x = torch.ones(2,2, requires_grad=True)
print(x.grad_fn)

y = x + 2
print(y)
print(y.grad_fn)

z = y * y * 3
out = z.mean()
print(z, out)

a = torch.randn(2, 2)
a = (a * 3) / (a - 1)
print(a.requires_grad)
a.requires_grad_()
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)

# torch求梯度，注意只能够计算叶子的梯度
out.backward()
print(x.grad)
print(y.retain_grad())
out2 = x.sum()
out2.backward()
print(x.grad)
#清零
x.grad.data.zero_()
out3 = x.sum()
out3.backward()
print(x.grad)

# 非标量求导的例子
v = torch.ones(2, 2)
x = torch.tensor([1, 2, 3, 4], dtype=torch.float, requires_grad=True)
y = 2 * x
z = y.view(2, 2)
print(z)
z.backward(v)
print(x.grad)

# 中断梯度追踪
x = torch.tensor(1.0, requires_grad=True)
y1 = x ** 2
with torch.no_grad():
    y2 = x ** 3
y3 = y1 + y2
print(x.requires_grad)
print(y1, y1.requires_grad)
print(y2, y2.requires_grad)
