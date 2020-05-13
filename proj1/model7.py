import torch
from torch import nn

class Net(nn.Module):
    # input is 64 x 64 x 3 image
    def __init__(self):
        super(Net, self).__init__()
        self.prep = nn.Sequential(
            nn.Linear(32 * 32 * 3, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 32 * 32 * 3),
            nn.Sigmoid()
        )
    def forward(self, x):
        # nezabudunut na sigmoid(x) a reshape((32,32,3))
        pred_img = self.prep(x)
        pred_img = pred_img.reshape((32, 32, 3))
        return pred_img


