""""
    Uncommented Net7 model from model.py in proj1 with removed exit statement.
    Returns a parameter tensor that is most similar(has highest cosine similarity) to input tensor.
"""

import torch
from torch import nn

class Net7(nn.Module):
    def __init__(self, shape):
        super(Net7, self).__init__()
        # 64 * 64 input
        self.fc1 = nn.Linear(64*64, 1000)
        self.list = nn.Parameter(torch.randn(1000, 1))
        # self.conv1 = nn.Conv2d(1, 6, 5)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(6, 16, 5)
        # self.fc1 = nn.Linear(16 * 4 * 4, 120)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = torch.Tensor(x)
        x = torch.flatten(x, 0)
        x = self.fc1(x)
        print(x.shape)
        x = torch.cosine_similarity(x, self.list, dim=0)
        print(x.shape)
        i = torch.argmax(x)
        x = self.list[i]
        print(x)
        # x = torch.sigmoid(x)
        # x = x.reshape(64, 64)
        return x