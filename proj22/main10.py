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
        self.seq = nn.Sequential(
            nn.Linear(2504, 5000),
            nn.ReLU(),
            nn.Linear(5000, 500),
            nn.
        )
        # self.linear = nn.Linear(2504, 5000)
        # self.param = nn.Parameter(torch.randint(1, (1, 1, 32, 32), dtype=float))

    def forward(self, img, locations):
        # f_img = torch.flatten(img, 0)
        f_loc = torch.flatten(locations)
        x = torch.cat((img, f_loc))

        st.write(x.shape)
        # x = torch.clamp(x, min=0, max=1)

        # x = F.interpolate(x, size=64, mode='bicubic') #mode='bicubic')#.permute(1, 2, 0)
        # x = torch.clamp(x, min=0, max=1)
        # x = torch.squeeze(x, dim=0)
        # x = x.permute(1, 2, 0)
        return x

    def upscale(self):
        x = self.param
        x = torch.squeeze(x, dim=0)
        x = torch.clamp(x, min=0, max=1)
        x = x.permute(1, 2, 0)
        return x

def load_img(path:str="game.png"):
    image = Image.open(path)
    # normalize
    img = np.array(image) / 255
    img = torch.tensor(img)
    return img

def make_patch_generator(image, shape, stride=1):
    x_max, y_max, c_max = image.shape
    for x in range(0, x_max - shape[0], stride):
        for y in range(0, y_max - shape[1], stride):
            yield image[x:x + shape[0], y:y + shape[1]], np.array([x, y, x + shape[0], y + shape[1]])


def run_app():
    net = Net()
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(net.parameters(), lr=0.01)

    st.title('image upscaler')
    img = load_img('game.png')

    st.write('Original image')
    orig_img_loc = st.image(img.numpy(), width=250, caption='original image: 64 x 64')

    KERNEL_SIZE = (25, 25)
    STRIDE = 10
    patch_generator = make_patch_generator(img.numpy(), KERNEL_SIZE, STRIDE)
    slid_win_loc = st.empty()
    for patch, location in patch_generator:
        sleep(0.1)
        slid_win_loc.image(patch, width=250, caption = f'sliding window patch at \nlocation: {location}\nshape: {patch.shape}')
        out = net(torch.tensor(patch), torch.tensor(location))
        # st.write(patch)

    # loc = [0.26, 0.1]
    # out = net(torch.tensor(loc, dtype=float), img)
    # upsam_img_loc = st.image(out.detach().numpy(), width=250, caption='Upsampled and Downsampled')
    # orig_img_loc2 = st.image(img.numpy(), width=640, caption='original image: 64 x 64')
    # upscaled = net.upscale()
    # upscaled_loc = st.image(upscaled.detach().numpy(), width=640, caption='Upscaled')
    # diff = torch.abs(out.detach() - img).numpy()
    # diff_img_loc = st.image(diff, width=250, caption=f'DIFF:{np.sum(diff)}')
    # # loss_loc = st.write('LOSS:?')
    #
    # while True:
    #     sleep(0.1)
    #     optimizer.zero_grad()
    #     out = net(torch.tensor([1.0], dtype=float))
    #     loss = criterion(out, img)
    #     loss.backward()
    #     optimizer.step()
    #     upsam_img_loc.image(out.detach().numpy(), width=250, caption='Upsampled and Downsampled')
    #     upscaled = net.upscale()
    #     upscaled_loc.image(upscaled.detach().numpy(), width=640, caption='Upscaled')
    #     diff = torch.abs(out.detach() - img).numpy()
    #     diff_img_loc.image(diff, width=250, caption=f'DIFF:{np.sum(diff)}')

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
