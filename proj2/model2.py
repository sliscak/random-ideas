import torch
from torch import nn

class Fusher(nn.Module):
    # input is 64 x 64 x 3 image
    def __init__(self):
        super(Fusher, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 32 * 32 * 3),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.block(x)
        return x

class Cleaner(nn.Module):
    # input is 64 x 64 x 3 image
    def __init__(self):
        super(Cleaner, self).__init__()
        self.block1 = nn.Sequential(
            nn.Linear(32*32*3, 1000),
            nn.LeakyReLU(),
            nn.Linear(1000, 500),
            nn.LeakyReLU(),
            nn.Linear(500, 32*32*3),
            nn.LeakyReLU()
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=6, padding=3, bias=True),
            nn.LeakyReLU(),
            nn.Conv2d(3, 3, kernel_size=6, padding=3, bias=True),
            nn.LeakyReLU(),
            nn.Conv2d(3, 3, kernel_size=6, padding=3, bias=True),
            nn.LeakyReLU(),
            nn.Conv2d(3, 3, kernel_size=6, padding=3, bias=True),
            nn.LeakyReLU(),
            nn.Conv2d(3, 3, kernel_size=6, padding=3, bias=True),
            nn.LeakyReLU(),
            nn.Conv2d(3, 3, kernel_size=6, padding=3, bias=True),
            nn.LeakyReLU(),
            nn.Conv2d(3, 3, kernel_size=7, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.p = nn.Parameter(torch.Tensor([1]))

    def forward(self, x):
        x = torch.flatten(x)
        x = self.block1(x)
        x = x.reshape((32, 32, 3))
        # x = self.block(x)
        # x = x.reshape((32, 32, 3))
        # INPUT: HxWxC -> 0,1,2
        # PERMUTED: CxHxW -> 2,0,1
        x = x.permute((2,0,1))
        x = torch.unsqueeze(x, dim=0)
        x = self.block2(x)
        x = x.squeeze(0)
        x = x.permute((1,2,0))
        #self.block[0].weight.data = self.block[0].weight.data - ((torch.rand_like(self.block[0].weight)*2)-1)
        # print(x.shape)
        # x = x.squeeze(0)
        # print(x.shape)
        return x

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

    def forward(self, x):
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

    def forward(self, x):
        return x