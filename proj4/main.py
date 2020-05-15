# Test a dynamic neural network with nearly unlimited capacity??
from proj4.model import DynamicNet
from time import sleep
import torch

dynet = DynamicNet()

while True:
    x_train = torch.rand((1,))
    output = dynet(x_train)
    print(f'Input: {x_train.detach()}\nOutput: {output.detach()}')
    sleep(1)