import torch
from models import Net, Net2, Net3, Net4
net = Net(3, 10)
t = torch.tensor([1,2,3])
out = net(t.detach())
print(out)

net2 = Net2(3, 10)
t = torch.tensor([1,2,3])
out = net2(t.detach())
print(out)

net3 = Net3(3, 10)
t = torch.tensor([1,2,3])
out = net3(t.detach())
print(out)

net4 = Net4(3, 10)
t = torch.tensor([1,2,3])
out = net4(t.detach())
print(out)