""""
    A variant of Adaptive Time Neural Network
"""

import torch
from torch import nn


class AdaptNet(nn.Module):
    def __init__(self, max_depth = 100):
        super(AdaptNet, self).__init__()
        self.max_depth = max_depth
        self.blocks = nn.ModuleList([nn.Sequential(
            nn.Linear(1, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.ReLU()
        ) for _ in range(max_depth)])
        self.classifiers = nn.ModuleList([nn.Sequential(
            nn.Linear(1, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.Sigmoid()
        ) for _ in range(max_depth)])

    def forward(self, x):
        confidence = 0
        output = None
        c = 0
        grow = False
        while confidence < 1:
            output = self.blocks[c](x)
            confidence += self.classifiers[c](output)
            c += 1
            print(f'Confidence: {confidence}')
            print(f'Executed Cycles: {c}')
            if c >= self.max_depth:
                print(f'Reached max depth/cycles: {c} from {self.max_depth}')
                break
        return output
