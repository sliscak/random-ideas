import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from time import sleep


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.param = nn.Parameter(torch.randn((64, 64, 3)))
        # self.conv1 = nn.Conv2d(1, 6, 3)
        # self.conv2 = nn.Conv2d(6, 16, 3)
        # # an affine operation: y = Wx + b
        # self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        permuted = x.permute(2, 0, 1)
        x = x * self.param
        x = torch.clamp(x, min=0, max=1)
        # Max pooling over a (2, 2) window
        # x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # # If the size is a square you can only specify a single number
        # x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # x = x.view(-1, self.num_flat_features(x))
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        return x

def updown(img):
    permuted = img.permute(2, 0, 1)
    upsampled = F.interpolate(permuted, size=64 * 5, mode='linear')

    # inject noise
    upsampled = torch.clamp(((torch.rand(upsampled.shape)*2)-1) * 0.1 + upsampled, min=0, max=1)

    downsampled = F.interpolate(upsampled, size=64).permute(1, 2, 0)
    return downsampled

def run_app():
    net = Net()

    st.title('image upscaler')
    image = Image.open('image.png')
    img = np.array(image) / 255
    img = torch.tensor(img)

    out = net(img)
    st.write(out.shape)
    # img = torch.unsqueeze(torch.tensor(img), 0)
    st.write('Original image')
    orig_img_loc = st.image(img.numpy(), width=250, caption='original image: 64 x 64')
    # st.write(f'Image shape: {img.shape}')
    # img = img.view((1,-1, 64, 64))
    # img = img.permute(0, 3, 1, 2)
    # st.write(f'Image shape: {img.shape}')
    # upsampled = F.interpolate(img.permute(0, 3, 1, 2), size= 64*3, mode='linear')#.permute(0, 2, 3, 1)
    # upsampled = F.interpolate(img.permute(2, 0, 1), size=64*5, mode='linear')#.permute(0, 2, 3, 1)

    # downsampled = F.interpolate(upsampled, 64).permute(0, 2, 3, 1)
    # downsampled = F.interpolate(upsampled, size=64).permute(1, 2, 0)

    # st.write(f'Upsampled Image shape: {downsampled.shape}')
    downsampled = updown(img)
    upsam_img_loc = st.image(downsampled.numpy(), width=250, caption='Upsampled and Downsampled')
    diff = torch.abs(downsampled - img).numpy()
    diff_img_loc = st.image(diff, width=250, caption=f'DIFF:{np.sum(diff)}')

    orig = img * 1
    while True:
        sleep(3)
        # img = torch.sigmoid()
        # img = torch.rand(img.shape) * orig
        # how to inject random noise?
        # img = torch.clamp(((torch.rand(img.shape)*2)-1) * 0.1 + orig, min=0, max=1)
        orig_img_loc.image(img.numpy(), width=250, caption='original image: 64 x 64')
        downsampled = updown(img)
        upsam_img_loc.image(downsampled.numpy(), width=250, caption='Upsampled and Downsampled')
        diff = torch.abs(downsampled - img).numpy()
        diff_img_loc.image(diff, width=250, caption=f'DIFF:{np.sum(diff)}')
     # = st.image(img*0.5, width=250)
    # img_loc.image(img*0, width=250)
    # orig_img_placeholder = st.empty()
    # orig_img_placeholder.image(data, width=250)


if __name__ == '__main__':
    run_app()
