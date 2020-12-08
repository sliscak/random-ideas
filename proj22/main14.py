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
    cuda = torch.device('cuda')
    cpu = torch.device('cpu')
    net = Net()
    net.to(cuda)
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(net.parameters(), lr=0.00001)

    st.title('image upscaler')
    img = load_img('game.png')

    st.write('Original image')
    orig_img_loc = st.image(img.numpy(), width=250, caption='original image: 64 x 64')

    KERNEL_SIZE = (25, 25)
    STRIDE = 150
    # patch_generator = make_patch_generator(img.numpy(), KERNEL_SIZE, STRIDE)
    slid_win_loc = st.empty()
    out_loc = st.empty()
    patch_generator = make_patch_generator(img.numpy(), KERNEL_SIZE, STRIDE)
    peace = [next(patch_generator) for x in range(20)]
    # output_data = st.empty()
    glob_loss_chart = st.empty()
    progress_bar = st.empty()
    slider = st.slider("SPEED", min_value=0.0, max_value=1.0)
    button = st.button('Click')
    slowdown = False
    losses = deque()
    col1, col2, col3 = st.beta_columns([3, 1])
    while True:
        # patch_generator = make_patch_generator(img.numpy(), KERNEL_SIZE, STRIDE)
        glob_loss = torch.tensor([0.0], device=cuda)
        # progress_bar.progress(0)
        optimizer.zero_grad()
        for patch, location in peace:
            # i += 1
            # progress_bar.progress(i * 10)
            # if (1 - slider) != 0:
            #     sleep(1 - slider)
            slid_win_loc.image(patch, width=250, caption = f'sliding window patch at \nlocation: {location}\nshape: {patch.shape}')
            out = net(torch.tensor(location, device=cuda).float())
            # output_data.table(out.cpu().detach().numpy())
            loss = criterion(out, torch.tensor(patch, device=cuda).float())
            glob_loss += loss
            if loss.cpu().detach() < 0.00001:
                slowdown = True
            if slowdown:
                sleep(2)

            out_loc.image(out.cpu().detach().numpy(), width=250, caption=f'LOSS: {loss.detach()}')
        losses.append(glob_loss.cpu().detach().numpy())
        glob_loss_data = pd.DataFrame(losses, columns=['global loss'])
        glob_loss_chart.line_chart(glob_loss_data)
        glob_loss.backward()
        optimizer.step()


if __name__ == '__main__':

    run_app()
