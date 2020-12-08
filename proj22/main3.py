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
        self.param = nn.Parameter(torch.randn((1, 3, 1024, 1024))* 1)

    def forward(self, x):
        x = x.permute(2, 0, 1)
        x = torch.unsqueeze(x, dim=0)
        # st.write(x.shape)

        # shape=(1, 3, 1024, 1024)
        x = F.interpolate(x, scale_factor=16, mode='bicubic')
        x = x + self.param
        x = torch.clamp(x, min=0, max=1)

        # st.write(x.shape)
        x = F.interpolate(x, size=64, mode='bicubic')#.permute(1, 2, 0)
        x = torch.clamp(x, min=0, max=1)
        x = torch.squeeze(x, dim=0)
        x = x.permute(1, 2, 0)
        return x

    def upscale(self, x):
        x = x.permute(2, 0, 1)
        x = torch.unsqueeze(x, dim=0)
        # shape=(1, 3, 1024, 1024)
        x = F.interpolate(x, scale_factor=16, mode='bicubic')
        x = x + self.param
        x = torch.clamp(x, min=0, max=1)

        x = torch.squeeze(x, dim=0)
        x = x.permute(1, 2, 0)
        return x

def updown(img):
    permuted = img.permute(2, 0, 1)
    upsampled = F.interpolate(permuted, size=64 * 5, mode='linear')

    # inject noise
    upsampled = torch.clamp(((torch.rand(upsampled.shape)*2)-1) * 0.005 + upsampled, min=0, max=1)

    downsampled = F.interpolate(upsampled, size=64).permute(1, 2, 0)
    return downsampled


def load_img(path:str="image.png"):
    image = Image.open(path)
    # normalize
    img = np.array(image) / 255
    img = torch.tensor(img)
    return img


def run_app():
    net = Net()
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(net.parameters(), lr=0.01)

    st.title('image upscaler')
    img = load_img('image.png')

    st.write('Original image')
    orig_img_loc = st.image(img.numpy(), width=250, caption='original image: 64 x 64')
    out = net(img)
    upsam_img_loc = st.image(out.detach().numpy(), width=250, caption='Upsampled and Downsampled')
    orig_img_loc2 = st.image(img.numpy(), width=640, caption='original image: 64 x 64')
    upscaled = net.upscale(img)
    upscaled_loc = st.image(upscaled.detach().numpy(), width=640, caption='Upscaled')
    diff = torch.abs(out.detach() - img).numpy()
    diff_img_loc = st.image(diff, width=250, caption=f'DIFF:{np.sum(diff)}')
    # loss_loc = st.write('LOSS:?')

    while True:
        sleep(0.1)
        optimizer.zero_grad()
        out = net(img)
        loss = criterion(out, img)
        loss.backward()
        optimizer.step()
        upsam_img_loc.image(out.detach().numpy(), width=250, caption='Upsampled and Downsampled')
        upscaled = net.upscale(img)
        upscaled_loc.image(upscaled.detach().numpy(), width=640, caption='Upscaled')
        diff = torch.abs(out.detach() - img).numpy()
        diff_img_loc.image(diff, width=250, caption=f'DIFF:{np.sum(diff)}')

    # orig = img * 1
    # while True:
    #     sleep(3)
    #
    #     orig_img_loc.image(img.numpy(), width=250, caption='original image: 64 x 64')
    #     downsampled = updown(img)
    #     upsam_img_loc.image(downsampled.numpy(), width=250, caption='Upsampled and Downsampled')
    #     diff = torch.abs(downsampled - img).numpy()
    #     diff_img_loc.image(diff, width=250, caption=f'DIFF:{np.sum(diff)}')


if __name__ == '__main__':
    run_app()
