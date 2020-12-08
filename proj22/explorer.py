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
        # n = 10
        self.keys = nn.Parameter(torch.randn(50, 4))
        self.keys2 = nn.Parameter(torch.randn(50, 4))

        self.values = nn.Parameter(torch.randn(50, 2500))
        self.values2 = nn.Parameter(torch.randn(50, 2500))


    def forward(self, locations):
        # f_img = torch.flatten(img, 0)
        key = torch.flatten(locations)
        # st.write(key.shape)
        # attention = torch.sigmoid(torch.matmul(self.keys, key))
        attention = torch.matmul(self.keys, key)
        attention = torch.softmax(attention, 0)
        # st.write(attention.shape)
        # st.write(self.keys.shape)
        # out = torch.matmul(attention, self.values)
        out = torch.matmul(attention, self.values)
        # out = torch.relu(out)

        # attention = torch.sigmoid(torch.matmul(self.keys2, out))
        # # out = torch.matmul(attention, self.values2)
        # out = torch.matmul(attention, self.keys2)
        out = torch.sigmoid(out)
        # st.write(out.shape)
        out = torch.reshape(out, (25, 25, 4))
        return out


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
