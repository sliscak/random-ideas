""""
    A dynamic neural network with nearly unlimited capacity??
"""

import torch
from torch import nn


class DynamicNet(nn.Module):
    def __init__(self):
        super(DynamicNet, self).__init__()
        self.blocks = nn.ModuleList([nn.Sequential(
            nn.Linear(1, 3),
            nn.ReLU(),
            nn.Linear(3, 3),
            nn.ReLU(),
            nn.Linear(3, 1),
            nn.ReLU()
        )])
        self.classifiers = nn.ModuleList([nn.Sequential(
            nn.Linear(1, 3),
            nn.ReLU(),
            nn.Linear(3, 3),
            nn.ReLU(),
            nn.Linear(3, 1),
            nn.Sigmoid()
        )])

    def forward(self, x):
        confidence = 0
        output = None
        c = 0
        max_depth = 100
        grow = False
        while confidence < 1:
            try:
                output = self.blocks[c](x)
            except IndexError:
                print(f'Creating new Block')
                self.blocks.append(
                    nn.Sequential(
                        nn.Linear(1, 3),
                        nn.ReLU(),
                        nn.Linear(3, 3),
                        nn.ReLU(),
                        nn.Linear(3, 1),
                        nn.ReLU()
                    )
                )
                output = self.blocks[c](x)
            try:
                confidence += self.classifiers[c](output)
            except IndexError:
                print(f'Creating new Classifier')
                self.classifiers.append(
                    nn.Sequential(
                        nn.Linear(1, 3),
                        nn.ReLU(),
                        nn.Linear(3, 3),
                        nn.ReLU(),
                        nn.Linear(3, 1),
                        nn.Sigmoid()
                    )
                )
                confidence += self.classifiers[c](output)
                grow = True
            c += 1
            print(f'Confidence: {confidence}')
            print(f'Executed Cycles: {c}')
            if c >= max_depth:
                print(f'Reached max depth/cycles: {max_depth}')
                break
        return output, grow
