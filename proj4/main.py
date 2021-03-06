# Test a dynamic neural network with nearly unlimited capacity??
from proj4.model import DynamicNet
from time import sleep
import torch

dynet = DynamicNet()
optimizer = torch.optim.Adam(params=dynet.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()

c = 0
while True:
    c += 1
    x_train = torch.rand((1,))
    output, grow = dynet(x_train)
    if grow:
        optimizer = torch.optim.Adam(params=dynet.parameters(), lr=0.01)
    loss = criterion(output, x_train.detach())
    print(f'Input: {x_train.detach()}\nOutput: {output.detach()}\nLoss: {loss.detach()}')
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    sleep(1)
    # print(optimizer.param_groups)
    # exit()
    # if c >= 2:
    #     print(len(list(dynet.parameters())))
    #     exit()