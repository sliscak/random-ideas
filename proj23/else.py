import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import os
# import kornia
from PIL import Image
from time import sleep
from collections import deque
from new_datasets import WikiDataset
from collections import Counter, namedtuple

class PavianX(nn.Module):


    def __init__(self, output_size=1):  # reset=False
        super().__init__()
        self.m = nn.Parameter(torch.ones(1))

    def forward(self, x):
        # x = self.m * x
        x = self.m * x
        if x <= 0.5 :
            print('LESS')
        else:
            x -= 10
            print("MORE")
        return x

net = PavianX()

criterion = nn.MSELoss()
optimizer = optim.AdamW(net.parameters(), lr=0.005)
while True:
    optimizer.zero_grad()
    out = net(torch.tensor(torch.ones(1)))
    y = torch.zeros(1)
    loss = criterion(out, y.clone().detach())
    loss.backward()
    optimizer.step()
    print(loss)
    # def configure_optimizers(self):
    #     optimizer = AdamW(self.parameters(), lr=0.003)#lr=0.0001)#lr=1e-4)
    #     return optimizer
    #
    # def training_step(self, batch, batch_idx):
    #     x_batch, y_batch = batch
    #     loss = torch.tensor([0.0])
    #     for i in range(len(x_batch)):
    #         x,y = x_batch[i], y_batch[i]
    #         loss += F.mse_loss(self(x), y)
    #     print(f'\n{loss.detach()}\n')
    #     # self.log('my_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    #     # x, y = batch
    #     # x = x[0]
    #     # y = y[0]
    #     # breakpoint()
    #     # self(x)
    #     # loss = F.mse_loss(self(x), y)
    #     # loss = F.mse_loss(self(x_batch), y_batch)
    #     # loss = F.cross_entropy(self(x_batch), y_batch)
    #     return pl.TrainResult(loss)
