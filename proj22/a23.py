import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import os
import kornia
from PIL import Image
from time import sleep
from collections import deque

class ImageDataset(Dataset):
    def __init__(self, path='', testing=False):
        self.image_folder_path = path
        self.data = []

        # TODO: Load on the fly
        for root, dirs, files in os.walk(path, topdown=False):
            for image_name in files:
                if '.JPEG' in image_name:
                    image_path = self.image_folder_path + image_name
                    image = load_img(image_path)
                    if len(image.shape) < 3:
                        continue
                    self.data.append(image)
                    break
                    # st.write('WORKING!')
        if not testing:
            self.data = self.data[0:100]
    # def load_images(self, path):

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        x = image
        y = image
        sample = x, y
        return sample


class LongConv(nn.Module):
    def __init__(self, size=1, keysize=None):
        super(LongConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=3, groups=1, kernel_size=(5, 5), stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, groups=1, kernel_size=(5, 5), stride=1, padding=2)
        self.conv3 = nn.Conv2d(in_channels=3, out_channels=3, groups=1, kernel_size=(5, 5), stride=1, padding=2)
        self.keys = nn.Parameter(torch.randn((size, keysize)))
        self.values = nn.Parameter(torch.randn((size, 3 * 3 * 5 * 5)))


    def forward(self, x):
        key = self.conv(x)
        # key = self.conv3(self.conv2(self.conv(x)))
        # key = torch.sigmoid(key)
        # key = F.interpolate(key, (32, 32))
        key = key.flatten()
        # key = torch.tanh(key)
        attention = torch.matmul(self.keys, key)
        attention = torch.softmax(attention, 0)
        attention = torch.reshape(attention, (-1, 1))

        kernel = self.values * attention
        kernel = torch.sum(kernel, 0)
        kernel = torch.reshape(kernel, (3, 3, 5, 5))

        out = torch.conv_transpose2d(x, weight=kernel, stride=3, padding=0)

        return out

class LongMem(nn.Module):
    def __init__(self, size=1, keysize=None, inch=3, outch=3,):
        super(LongMem, self).__init__()
        self.inch, self.outch = inch, outch
        self.keys = nn.Parameter(torch.randn((size, keysize)))
        self.values = nn.Parameter(torch.randn((size, inch * outch * 5 * 5)))

    def forward(self, x):
        key = 1 * x
        # key = F.interpolate(x, (32, 32))
        # key = torch.sigmoid(key)
        key = key.flatten()
        # key = torch.tanh(key)
        attention = torch.matmul(self.keys, key)
        attention = torch.softmax(attention, 0)
        attention = torch.reshape(attention, (-1, 1))

        kernel = self.values * attention
        kernel = torch.sum(kernel, 0)
        kernel = torch.reshape(kernel, (self.inch, self.outch, 5, 5))

        # out = torch.conv_transpose2d(x, weight=kernel, stride=2, padding=2)
        out = torch.conv_transpose2d(x, weight=kernel, stride=1)
        return out

class Mem(nn.Module):
    def __init__(self, size=1, keysize=None, outputsize=None):
        super(LongMem, self).__init__()
        self.keys = nn.Parameter(torch.randn((size, keysize)))
        self.values = nn.Parameter(torch.randn((size, outputsize)))

    def forward(self, x):
        key = F.interpolate(x, (32, 32))
        # key = torch.sigmoid(key)
        key = key.flatten()
        attention = torch.matmul(self.keys, key)
        attention = torch.softmax(attention, 0)
        attention = torch.reshape(attention, (-1, 1))

        value = self.values * attention
        value = torch.sum(value, 0)
        out = torch.reshape(value, (self.inch, self.outch, 5, 5))
        return out

class Looper(nn.Module):
    def __init__(self, size=1, keysize=None):
        super(Looper, self).__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=3, groups=1, kernel_size=(5, 5), stride=2)
        self.keys = nn.Parameter(torch.randn((size, keysize)))
        self.values = nn.Parameter(torch.randn((size, 3 * 3 * 5 * 5)))

    def forward(self, x, y):
        key = self.conv(x)
        key = torch.sigmoid(key)
        key = F.interpolate(key, (32, 32))
        key = key.flatten()
        # key = torch.tanh(key)
        attention = torch.matmul(self.keys, key)
        attention = torch.softmax(attention, 0)
        attention = torch.reshape(attention, (-1, 1))

        kernel = self.values * attention
        kernel = torch.sum(kernel, 0)
        kernel = torch.reshape(kernel, (3, 3, 5, 5))
        out = torch.conv_transpose2d(x, weight=kernel, stride=2, padding=2)
        # out = torch.sigmoid(out)
        # out = F.interpolate(out, size=(64, 64))
        it = torch.tensor([0])
        # while 1/kornia.psnr_loss(out, y, max_val=1.0) > 0.0:
        while F.mse_loss(out, y) > 0.0:
            # st.write(f'ITERATION: {it} ol: {ol}')
            it += 1
            key = self.conv(out)
            key = torch.sigmoid(key)
            key = F.interpolate(key, (32, 32))
            key = key.flatten()
            # key = torch.tanh(key)
            attention = torch.matmul(self.keys, key)
            attention = torch.softmax(attention, 0)
            attention = torch.reshape(attention, (-1, 1))

            kernel = self.values * attention
            kernel = torch.sum(kernel, 0)
            kernel = torch.reshape(kernel, (3, 3, 5, 5))
            out = torch.conv_transpose2d(out, weight=kernel, stride=2, padding=2)
            if it >= 5:
                break
        # st.write(f'ITERATIONS: {it}')
        return out

upped_image = st.empty()

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.lk1 = LongConv(10, 12288)
        self.lk2 = LongConv(10, 108300)
        self.lk3 = LongConv(100, 3072)
        self.lk4 = LongConv(50, 3072)
        self.lk5 = LongConv(50, 3072)
        self.lk6 = LongConv(50, 3072)

        self.lc = Looper(100, 3072)
        # self.lm = LongMem(1, 3072)
        self.lm = LongMem(10, 12288, 3, 3)
        self.lm2 = LongMem(10, 13872, 3, 3)
        self.lm3 = LongMem(10, 15552, 3, 3)

        self.conv = nn.Conv2d(3, 3, (3, 3), stride=1)
        self.conv2 = nn.Conv2d(3, 3, (3, 3), stride=1)
        self.conv3 = nn.Conv2d(3, 3, (3, 3), stride=1)

        self.tconv = nn.ConvTranspose2d(3, 3, (10, 10))
        self.tconv2 = nn.ConvTranspose2d(3, 3, (10, 10))

        self.m = Mem()
    def forward(self, input):
        input = F.interpolate(input, size=(128, 128))
        input = (2 * input) - 1
        out = self.m(input)
        # st.write(f'INPUT: {input.shape}')
        # out = self.lk1(input)
        # out = self.conv(input)
        # out = torch.relu(out)
        # out = self.conv2(out)
        # out = torch.relu(out)
        # out = self.conv3(out)
        # image = out.permute(0, 3, 1, 2)
        # out = self.lk2(out)
        # st.write(f'SHAPE: {out.shape}')
        out = torch.sigmoid(out)
        image = out.permute(0, 2, 3, 1)
        upped_image.image(image.detach().cpu().numpy(), width=250, caption='upscaled')
        out = F.interpolate(out, size=(64, 64))
        return out

def load_img(path:str="image.png"):
    image = Image.open(path)
    # normalize
    img = np.array(image) / 255
    # img = torch.tensor(img)
    return img

def process(image):
    pass

def run_app():
    # dataset_textbox = st.sidebar.text_input('dataset path', value='C:\\Users\\Admin\\Downloads\\i\\n01514859\\')

    DATASET_PATH = st.text_input('DATASET PATH', value='C:\\Users\\Admin\\Downloads\\i\\n01514859\\')
    epoch_loc = st.empty()
    prog_bar = st.empty()
    loss_loc = st.empty()
    global_loss_loc = st.empty()
    loss_chart = st.empty()
    glob_loss_chart = st.empty()
    row0 = st.empty()
    row1 = st.empty()
    row2 = st.empty()
    row3 = st.empty()
    row4 = st.empty()
    row5 = st.empty()

    # st.stop()
    PATH = "upscaler.pt"
    net = Net()
    # too lazy to detect if the file exits.
    try:
        net.load_state_dict(torch.load(PATH))
        st.write('MODEL LOADED!')
    except Exception:
        pass
    cuda = torch.device('cuda')
    net.to(cuda)
    # criterion = nn.CrossEntropyLoss()
    # criterion = nn.MSELoss()
    criterion = kornia.losses.PSNRLoss(1.0)
    LEARNING_RATE = 0.01
    optimizer = optim.AdamW(net.parameters(), lr=LEARNING_RATE)
    # st.title('image upscaler')
    img = load_img('image.png')

    losses = deque(maxlen=100)
    global_losses = deque(maxlen=100)
    EPOCHS = 500
    BATCH_SIZE = 1

    dataset = ImageDataset(path=DATASET_PATH)
    def collate_wrapper(samples):
        return samples

    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_wrapper)
    for epoch in range(EPOCHS):
        i = 1
        epoch_loc.write(f"EPOCH:\t{epoch}/{EPOCHS - 1}")
        global_loss = torch.tensor([0.0], device=cuda)
        optimizer.zero_grad()
        # TODO: confirm that shuffle works
        # --------------------
        for batch in train_loader:
            optimizer.zero_grad()
            loss = torch.tensor([0.0], device=cuda)
            for sample in batch:
                x, y = sample
                x = torch.tensor(x)
                x = torch.unsqueeze(x, 0)
                try:
                    image = x.permute(0, 3, 1, 2)
                    image = F.interpolate(image, size=(128, 128))
                    image = image.permute(0, 2, 3, 1)
                    row1.image(image.numpy(), width=250, caption='original image')
                except Exception:
                    break
                x = x.permute(0, 3, 1, 2)
                y = F.interpolate(x, size=(64, 64))
                x = F.interpolate(x, size=(32, 32))
                x = F.interpolate(x, size=(64, 64))
                row2.image(x.permute(0, 2, 3, 1).detach().numpy(), width=250, caption='Downsampled')
                prog_bar.progress(i / len(dataset))
                i += 1
                out = net(x.detach().cuda().float())
                diff = torch.abs(out.detach().cpu() - y.detach().cpu())
                diff_image = diff.permute(0, 2, 3, 1).numpy()
                row5.image(diff_image, width=250, caption='absolute difference')
                row3.image(out.permute(0, 2, 3, 1).detach().cpu().numpy(), width=250, caption='Reconstructed')
                loss = 1 / criterion(out, y.detach().cuda().float())
                # loss = criterion(out, y.detach().cuda().float())

                row4.write(f'LOSS: {loss.detach().cpu()}')
                # loss.backward()
                # optimizer.step()
                # st.stop()
            losses.append(loss.detach().cpu().numpy())
            loss_chart.line_chart(
                pd.DataFrame(losses, columns=['loss',])
            )
            global_loss += loss
            loss_loc.write(f"LOSS:\t{loss.detach().cpu()}")
            loss.backward()
            optimizer.step()
        global_loss_loc.write(f"GLOBAL LOSS:\t{global_loss.detach().cpu()}  \nGLOB AVERAGE LOSS:\t{global_loss.detach().cpu()/len(dataset)}")
        global_losses.append(global_loss.detach().cpu().numpy())
        glob_loss_chart.line_chart(
            pd.DataFrame(global_losses, columns=['global_loss', ])
        )
    try:
        torch.save(net.state_dict(), PATH)
        st.write('MODEL SAVED!')
    except Exception:
        pass

if __name__ == '__main__':
    run_app()
