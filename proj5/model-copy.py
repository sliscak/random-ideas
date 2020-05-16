""""
    A dynamic neural network with nearly unlimited capacity??
"""

import torch
from torch import nn

class DynamicNet(nn.Module):
    def __init__(self):
        super(DynamicNet, self).__init__()
        self.blocks = nn.ModuleList([])
        self.classifiers = nn.ModuleList([])

    def forward(self, x):
        confidence = 0
        output = None
        c = 0
        while confidence < 1:
            try:
                output = self.blocks[c](x)
            except IndexError:
                self.blocks.append(
                    nn.Sequential(
                        nn.Linear(1, 10),
                        nn.ReLU(),
                        nn.Linear(10, 10),
                        nn.ReLU(),
                        nn.Linear(10, 1),
                        nn.ReLU(model.py)
                    )
                )
            try:
                confidence += self.classifiers[c](output)
            except IndexError:
                self.classifiers.append(
                    nn.Sequential(
                        nn.Linear()
                    )
                )
            c += 1
        return x