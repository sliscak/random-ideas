import torch
from torch import nn


class RandomNet(nn.Module):
    def __init__(self):
        super(RandomNet, self).__init__()
        # self.block = nn.Sequential(
        #     nn.Linear(32, 3072),
        #     nn.Sigmoid(),
        # )
        self.block = nn.Sequential(
            nn.Linear(32, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 3072),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = torch.flatten(x)
        x = self.block(x)
        x = torch.reshape(x, (32, 32, 3))
        return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.p = nn.Parameter(torch.rand((32,)), requires_grad=True)

    def forward(self, x):
        x = x + self.p
        return x