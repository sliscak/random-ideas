""""
    Seeing what patterns the neural network has learned.
"""

import retro
# from PIL import ImageOps
from collections import deque
import streamlit as st
from skimage import data, transform
from skimage.color import rgb2gray
import scipy
import torch
import numpy as np
import torch
from time import time
import math
from time import sleep
from torch import nn
import torch

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(64*64, 2, bias=False)

    def forward(self, x):
        x = self.layer(x)
        x = torch.sigmoid(x)
        return x

model = Net()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.2)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.000003)
criterion = torch.nn.MSELoss()

window = [[col.empty() for col in st.beta_columns(7)] for x in range(50)]


env = retro.make(game='Airstriker-Genesis')
obs = env.reset()
for i in range(100000):
    # action = env.action_space.sample()
    # obs, rew, done, info = env.step(action)
    image = rgb2gray(obs)
    image = transform.resize(image, (64, 64))
    # print((np.min(image), np.max(image)))
    image_t = torch.tensor(image, dtype=torch.float).flatten()
    if i%2 == 0:
        y = torch.tensor([0.0, 1.0])
    else:
        y = torch.tensor([1.0, 0.0])
    y_pred = model(image_t)
    loss = criterion(y_pred, y)
    window[0][1].write(f'LOSS: {loss.detach()}')
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    window[0][0].image(image, caption='input', width=200)
    weight = model.layer.weight.detach()
    with torch.no_grad():
        w1 = weight[0].reshape((64, 64)).numpy() # this is the pattern(before normalization) from the first neuron from the linear layer of the network
        w2 = weight[1].reshape((64, 64)).numpy() # this is the pattern(before normalization) from the second neuron from the linear layer of the network

        # window[2][0].write(w1)
        # window[2][1].write(w2)

        w1 = w1 + np.abs(np.min(w1))
        w1 = w1 / np.max(w1)  # this is the pattern(after normalization) from the first neuron from the linear layer of the network


        w2 = w2 + np.abs(np.min(w2))
        w2 = w2 / np.max(w2) # this is the pattern(after normalization) from the second neuron from the linear layer of the network
        # w1 = np.clip(w1, 0, 1)
        # w2 = np.clip(w2, 0, 1)

        window[1][0].image(w1, caption='w1', width=100) # showing first pattern
        window[1][1].image(w2, caption='w2', width=100) # showing the second pattern
