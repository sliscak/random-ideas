import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from time import sleep
from collections import deque

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(4, 2199),
            nn.ReLU(),
            nn.Linear(2199, 2199),
            nn.ReLU(),
            nn.Linear(2199, 2199),
            nn.ReLU(),
            nn.Linear(2199, 2199),
            nn.ReLU(),
            nn.Linear(2199, 2199),
            nn.ReLU(),
            nn.Linear(2199, 2199),
            nn.ReLU(),
            nn.Linear(2199, 2199),
            nn.ReLU(),
            nn.Linear(2199, 25*25*4),
            nn.Sigmoid(),
        )
        # self.linear = nn.Linear(2504, 5000)
        # self.param = nn.Parameter(torch.randint(1, (1, 1, 32, 32), dtype=float))

    def forward(self, locations):
        # f_img = torch.flatten(img, 0)
        f_loc = torch.flatten(locations)
        x = self.seq(f_loc)
        x = x.reshape((25, 25, 4))
        # x = x.permute(2, 0, 1)
        return x
        # x = torch.cat((f_img, f_loc))

        # st.write(f_loc.shape)
        # x = torch.clamp(x, min=0, max=1)

        # x = F.interpolate(x, size=64, mode='bicubic') #mode='bicubic')#.permute(1, 2, 0)
        # x = torch.clamp(x, min=0, max=1)
        # x = torch.squeeze(x, dim=0)
        # x = x.permute(1, 2, 0)

    # def upscale(self):
    #     x = self.param
    #     x = torch.squeeze(x, dim=0)
    #     x = torch.clamp(x, min=0, max=1)
    #     # st.write(x.shape)
    #     x = x.permute(1, 2, 0)
    #     return x

def load_img(path:str="game.png"):
    image = Image.open(path)
    # normalize
    img = np.array(image) / 255
    img = torch.tensor(img)
    return img

def fun(x):
    return (x * 2) - 1

def make_patch_generator(image, shape, stride=1):
    x_max, y_max, c_max = image.shape
    for x in range(0, x_max - shape[0], stride):
        for y in range(0, y_max - shape[1], stride):
            yield image[x:x + shape[0], y:y + shape[1]], \
                  np.array([fun(x/x_max), fun(y/y_max), fun((x + shape[0])/x_max), fun((y + shape[1])/y_max)])


# @st.cache(suppress_st_warning=True)
def run_app():
    PATH = "state_dict_model.pt"

    cuda = torch.device('cuda')
    cpu = torch.device('cpu')
    net = Net()
    net.load_state_dict(torch.load(PATH))
    net.eval()
    net.to(cuda)

    X1_slider = st.slider('X1', min_value=-1.0, max_value=1.0)
    Y1_slider = st.slider('Y1', min_value=-1.0, max_value=1.0)
    X2_slider = st.slider('X2', min_value=-1.0, max_value=1.0)
    Y2_slider = st.slider('Y2', min_value=-1.0, max_value=1.0)

    location = [X1_slider, Y1_slider, X2_slider, Y2_slider]
    out = net(torch.tensor(location, device=cuda).float())
    st.image(out.cpu().detach().numpy(), width=250, caption=f'patch')
if __name__ == '__main__':
    run_app()
