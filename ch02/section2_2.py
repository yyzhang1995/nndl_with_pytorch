import torch

# 创建张量
print("创建和查看张量")
x = torch.empty(5, 3)
print(x)

x = torch.rand(5, 3)
x = torch.zeros(5, 3, dtype=torch.long)
x = torch.tensor([5.5, 3])
x = x.new_ones(5, 3, dtype=torch.float64)
print(x)

print(x.size())
print(x.shape)

# torch操作
print("torch操作")
y = torch.rand(5, 3)
print(x + y)
print(torch.add(x, y))

# torch索引
print("torch索引")
y = x[0, :]

#指定维度
print("torch改变维度")
y = x.view(15)
print(y.size())
z = x.view(-1, 5)
print(z.size())

# torch复制
# y = x.clone()
# y = x.item()

# torch线性代数
print("torch线性代数")
y = x.t()
print(y)

# torch广播

# torch内存开销
x = torch.tensor([5, 6])
y = torch.tensor([1, 2])
print("这种方式会使用新内存")
y = x + y
print("不会开新内存的方式")
y[:] = x + y

# torch与numpy相互转换
# .from_numpy
# .numpy

# 在GPU上使用tensor
if torch.cuda.is_available():
    device = torch.device("cuda")
    y = torch.ones_like(x, device=device)
    x = x.to(device)
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))