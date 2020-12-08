import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import kornia
from PIL import Image
from time import sleep


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # self.param = nn.Parameter(torch.randint(1, (1, 3, 1024, 1024), dtype=float))
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, groups=1, kernel_size=(5, 5), stride=2)
        self.conv = nn.Conv2d(in_channels=3, out_channels=1, groups=1, kernel_size=(5, 5), stride=2)

        self.keys = nn.Parameter(torch.randn((5,900)))
        self.values = nn.Parameter(torch.randn((5, 3*3*5*5)))

        self.deconvs = nn.Sequential(
            nn.ConvTranspose2d(3, 3, (10, 10), stride=2, padding=2),
            # nn.ReLU(),
            # nn.ConvTranspose2d(3, 3, (5, 5), stride=2, padding=2),
            # nn.ReLU(),
            # nn.ConvTranspose2d(3, 3, (5, 5), stride=2, padding=2),
            # nn.ReLU(),
            # nn.ConvTranspose2d(3, 3, (5, 5), stride=2, padding=2),
            # nn.ReLU(),
            # nn.ConvTranspose2d(3, 3, (5, 5), stride=2, padding=2),
            # nn.ReLU(),
            # nn.ConvTranspose2d(3, 3, (5, 5), stride=2, padding=2),
        )
        self.deconv1 = nn.ConvTranspose2d(3, 3, (5, 5), stride=2, padding=2)

        self.deconv2 = nn.ConvTranspose2d(3, 3, (5, 5), stride=2, padding=2)
        self.deconv3 = nn.ConvTranspose2d(3, 3, (5, 5), stride=2, padding=2)
        # self.deconv1 = nn.ConvTranspose2d(3, 3, (5, 5), stride=2)

    def forward(self, input):
        key = self.conv(input).flatten()
        attention = torch.matmul(self.keys, key)
        attention = torch.softmax(attention, 0)
        attention = torch.reshape(attention, (-1, 1))

        kernel = self.values * attention
        kernel = torch.sum(kernel, 0)
        kernel = torch.reshape(kernel, (3, 3, 5, 5))

        out = torch.conv_transpose2d(input, weight=kernel, stride=2, padding=2)
        out = torch.sigmoid(out)
        out = F.interpolate(out, size=(64, 64))
        # st.stop()
        return out
        #
        # # st.write(x.shape)
        # x = (2 * x) - 1
        # x = self.deconv1(x)
        # # x = torch.relu(x)
        # x = self.deconv2(x)
        # x = self.deconv3(x)
        # x = torch.sigmoid(x)
        # # x = self.conv1(x)
        # # x = torch.sigmoid(x)
        # # x = self.deconv2(x)
        # # x = torch.sigmoid(x)
        # x = F.interpolate(x, size=(64, 64))
        # x = torch.relu(x)
        # x = self.deconv2(x)
        # x = F.interpolate(x, size=(60, 60))
        # x = torch.relu(x)
        # x = self.deconv3(x)
        # x = torch.sigmoid(x)
        # x = torch.sigmoid(x)
        # x = F.interpolate(x, size=(64, 64))
        # x = self.deconv2(x)
        # x = torch.sigmoid(x)
        # x = F.interpolate(x, size=(64, 64))
        # x = self.deconv3(x)
        # x = torch.sigmoid(x)
        # x = F.interpolate(x, size=(64, 64))
        # x = torch.clamp(x, 0, 255)
        # x = torch.sigmoid(x)
        # x = self.deconvs(x)
        # x = self.conv1(x)
        # x = torch.unsqueeze(x, 0)
        # st.write(x.shape)
        # x = x * self.param
        # # x = torch.clamp(x, min=0, max=1)
        #
        # x = F.interpolate(x, size=(32, 32)) #mode='bicubic')#.permute(1, 2, 0)
        # x = torch.clamp(x, min=0, max=1)
        # x = torch.squeeze(x, dim=0)
        # x = x.permute(1, 2, 0)
        # x = torch.squeeze(x, 0)
        # x = x.permute(2, 1, 0)
        # x = torch.sigmoid(x)
        # x = F.interpolate(x, size=(64, 64))
        # st.write(x.shape)
        return x

    # def upscale(self):
    #     x = self.param
    #     x = torch.squeeze(x, dim=0)
    #     x = torch.clamp(x, min=0, max=1)
    #     x = x.permute(1, 2, 0)
    #     return x

def load_img(path:str="image.png"):
    image = Image.open(path)
    # normalize
    img = np.array(image) / 255
    # img = torch.tensor(img)
    return img

def process(image):
    pass

def run_app():
    net = Net()
    cuda = torch.device('cuda')
    net.to(cuda)
    # criterion = nn.CrossEntropyLoss()
    criterion = kornia.losses.PSNRLoss(1.0)
    optimizer = optim.AdamW(net.parameters(), lr=0.01)
    # st.title('image upscaler')
    img = load_img('image.png')
    # col1, col2, col3, col4 = st.beta_columns(4)
    line1 = st.empty()
    line2 = st.empty()
    line3 = st.empty()
    line4 = st.empty()
    line5 = st.empty()

    # line1.image(img, width=250, caption='original image')
    # st.write('Original image')
    # orig_img_loc = st.image(img.numpy(), width=250, caption='original image: 64 x 64')
    # out = net(torch.tensor([1.0], dtype=float))
    # upsam_img_loc = st.empty()#st.image(out.detach().numpy(), width=250, caption='Upsampled and Downsampled')
    # orig_img_loc2 = st.image(img.numpy(), width=640, caption='original image: 64 x 64')
    # # upscaled = net.upscale()
    # # upscaled_loc = st.image(upscaled.detach().numpy(), width=640, caption='Upscaled')
    # diff = torch.abs(out.detach() - img).numpy()
    # diff_img_loc = st.image(diff, width=250, caption=f'DIFF:{np.sum(diff)}')
    # # loss_loc = st.write('LOSS:?')

    while True:
        optimizer.zero_grad()
        x = torch.tensor(img)
        # torch.Size([1, 64, 64, 3])
        x = torch.unsqueeze(x, 0)
        # torch.Size([1, 64, 32, 32])
        line1.image(x.numpy(), width=250, caption='original image')
        x = x.permute(0, 3, 1, 2)
        y = 1 * x
        # torch.Size([1, 3, 64, 64])
        x = F.interpolate(x, size=(32, 32))
        x = F.interpolate(x, size=(64, 64))
        # torch.Size([1, 3, 32, 32])
        # st.write(x.shape)
        line2.image(x.permute(0, 2, 3, 1).detach().numpy(), width=250, caption='Downsampled')
        out = net(x.detach().cuda().float())
        diff = torch.abs(out.detach().cpu() - (F.interpolate(x, size=(64, 64))).detach().cpu())
        diff_image = diff.permute(0, 2, 3, 1).numpy()
        line5.image(diff_image, width=250)
        line3.image(out.permute(0, 2, 3, 1).detach().cpu().numpy(), width=250, caption='Reconstructed')
        loss = 1/criterion(out, y.detach().cuda().float())
        line4.write(f'LOSS: {loss.detach().cpu()}')
        loss.backward()
        optimizer.step()
        # st.write(out.shape)
        # st.stop()
        # loss.backward()
        # optimizer.step()
        # upsam_img_loc.image(out.detach().cpu().numpy(), width=250, caption='Upsampled and Downsampled')
        # upscaled = net.upscale()
        # upscaled_loc.image(upscaled.detach().numpy(), width=640, caption='Upscaled')
        # diff = torch.abs(out.detach() - img).numpy()
        # diff_img_loc.image(diff, width=250, caption=f'DIFF:{np.sum(diff)}')


if __name__ == '__main__':
    run_app()
