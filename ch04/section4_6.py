import torch
import torch.nn as nn

print(torch.cuda.is_available())

print(torch.cuda.device_count())

print(torch.cuda.current_device())

print(torch.cuda.get_device_name(0))

x = torch.tensor([1, 2, 3])
x = x.cuda(0)
print(x)
print(x.device)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

y = x ** 2
print(y)

net = nn.Linear(3, 1)
net.cuda()

print(list(net.parameters())[0].device)

x = torch.rand(2, 3).cuda()
print(net(x))