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
        self.classifiers = nn.ModuleList([nn.Sequential(
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.Sigmoid()
        ) for _ in range(max_depth)])

    def forward(self, x):
        x = self.first_block(x)
        confidence = 0
        c = 0
        while confidence < 1:
            x = self.blocks[c](x)
            confidence += self.classifiers[c](x)
            c += 1
            print(f'Confidence: {confidence}')
            print(f'Executed Cycles: {c}')
            if c >= self.max_depth:
                print(f'Reached max depth/cycles: {c} from {self.max_depth}')
                break
        x = self.last_block(x)
        return x
