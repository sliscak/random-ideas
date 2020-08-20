""""
    A variant of Adaptive Time Neural Network
"""
import torch
from torch import nn


class AdaptNet(nn.Module):
    def __init__(self, max_depth = 100):
        super(AdaptNet, self).__init__()
        self.max_depth = max_depth
        self.first_block = nn.Sequential(
            nn.Linear(1, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU()
        )
        self.last_block = nn.Sequential(
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.Sigmoid()
        )
        self.blocks = nn.ModuleList([nn.Sequential(
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU()
        ) for _ in range(max_depth)])
        # those classifiers calculate how much confidence there is in the output tensors of blocks
        self.classifiers = nn.ModuleList([nn.Sequential(
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.Sigmoid()
        ) for _ in range(max_depth)])

        # learnable treshold?
        # self.treshold = nn.Parameter(torch.Tensor([1]))
        self.treshold = torch.Tensor([1])

    def forward(self, x):
        x = self.first_block(x)
        confidence = 0
        c = 0
        while confidence < self.treshold:
            x = self.blocks[c](x)
            partial_confidence = self.classifiers[c](x)
            confidence += partial_confidence
            c += 1
            print(f'Confidence: {confidence}')
            print(f'Executed Cycles: {c}')
            print(f'Current Treshold: {self.treshold}')
            if c >= self.max_depth:
                print(f'Reached max depth/cycles: {c} from {self.max_depth}')
                break
            if partial_confidence <= 0:
                print('No improvement detected. Breaking from Loop.')
                break
        x = self.last_block(x)
        return x
