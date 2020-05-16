""""
    A dynamic neural network with nearly unlimited capacity??
"""

import torch
from torch import nn


class DynamicNet(nn.Module):
    def __init__(self, max_depth = 100):
        super(DynamicNet, self).__init__()
        self.max_depth = max_depth
        self.blocks = nn.ModuleList([nn.Sequential(
            nn.Linear(1, 3),
            nn.ReLU(),
            nn.Linear(3, 3),
            nn.ReLU(),
            nn.Linear(3, 1),
            nn.ReLU()
        ) for _ in range(max_depth)])
        self.classifiers = nn.ModuleList([nn.Sequential(
            nn.Linear(1, 3),
            nn.ReLU(),
            nn.Linear(3, 3),
            nn.ReLU(),
            nn.Linear(3, 1),
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
