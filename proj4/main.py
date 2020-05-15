# Test a dynamic neural network with nearly unlimited capacity??
from proj4.model import DynamicNet
from time import sleep
import torch

dynet = DynamicNet()
optimizer = torch.optim.SGD(params=dynet.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()

while True:
    x_train = torch.rand((1,))
    output = dynet(x_train)
    loss = criterion(output, x_train.detach())
    print(f'Input: {x_train.detach()}\nOutput: {output.detach()}\nLoss: {loss.detach()}')
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    sleep(1)