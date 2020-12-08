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
                    self.data.append(image)
                    break
                    # st.write('WORKING!')
        if not testing:
            self.data = self.data[0:10]
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
        self.conv = nn.Conv2d(in_channels=3, out_channels=3, groups=1, kernel_size=(5, 5), stride=2)
        self.keys = nn.Parameter(torch.randn((size, keysize)))
        self.values = nn.Parameter(torch.randn((size, 3 * 3 * 5 * 5)))

    def forward(self, x):
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

        return out

class Looper(nn.Module):
    def __init__(self, size=1, keysize=None):
        super(Looper, self).__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=3, groups=1, kernel_size=(5, 5), stride=2)
        self.keys = nn.Parameter(torch.randn((size, keysize)))
        self.values = nn.Parameter(torch.randn((size, 1)))

    def forward(self, x):
        key = self.conv(x)
        key = torch.sigmoid(key)
        key = F.interpolate(key, (32, 32))
        key = key.flatten()
        # key = torch.tanh(key)
        attention = torch.matmul(self.keys, key)
        attention = torch.softmax(attention, 0)
        attention = torch.reshape(attention, (-1, 1))

        kernel = self.values * attention
        out = torch.sum(kernel, 0)
        return out


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.lk1 = LongConv(100, 3072)
        self.lk2 = LongConv(100, 3072)
        self.lk3 = LongConv(100, 3072)
        self.lk4 = LongConv(50, 3072)
        self.lk5 = LongConv(50, 3072)
        self.lk6 = LongConv(50, 3072)

        self.lc = Looper(100, 3072)

    def forward(self, input):
        input = (2 * input) - 1
        # out1 = self.lk1(input)
        # # out1 = torch.relu(out1)
        # out2 = self.lk2(out1)
        # # out2 = torch.relu(out2)
        # out3 = self.lk3(out2)
        # out3 = torch.relu(out3)
        out = self.lk1(input)
        ix = torch.tensor([0.0], device='cuda')
        ix += torch.sigmoid(self.lc(out))
        it = 0
        while ix < 1:
            it += 1
            out = self.lk1(out)
            ix += torch.sigmoid(self.lc(out))
            if it >= 10:
                ix = 1
            st.write(f'LOOPS: {it}')
        # out3 = self.lk1(out2)
        # out2 = self.lk2(out1)
        # out3 = self.lk3(input)
        # out4 = self.lk4(out3)
        # out5 = self.lk5(input)
        # out = out2 + out4 #+ out4 + out5
        # out4 = self.lk3(out3)
        # out4 = torch.relu(out4)
        # out5 = self.lk4(out4)
        # out6 = self.lk5(out5)
        # out = self.lk4(out)
        out = torch.sigmoid(out)
        out = F.interpolate(out, size=(128, 128))
        # out = torch.sigmoid(out)
        # out1 = F.interpolate(out, size=(64, 64))
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
    epoch_loc = st.empty()
    prog_bar = st.empty()
    loss_loc = st.empty()
    global_loss_loc = st.empty()
    loss_chart = st.empty()
    glob_loss_chart = st.empty()
    line0 = st.empty()
    line1 = st.empty()
    line2 = st.empty()
    line3 = st.empty()
    line4 = st.empty()
    line5 = st.empty()

    PATH = "upscaler.pt"
    net = Net()
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
    optimizer = optim.AdamW(net.parameters(), lr=0.005)
    # st.title('image upscaler')
    img = load_img('image.png')

    losses = deque(maxlen=100)
    global_losses = deque(maxlen=100)
    EPOCHS = 500
    BATCH_SIZE = 10
    dataset = ImageDataset(path='C:\\Users\\Admin\\Downloads\\i\\n01514859\\')
    def collate_wrapper(samples):
        return samples

    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_wrapper)
    for epoch in range(EPOCHS):
        i = 1
        epoch_loc.write(f"EPOCH:\t{epoch}/{EPOCHS - 1}")
        global_loss = torch.tensor([0.0], device=cuda)
        optimizer.zero_grad()
        right = 0
        wrong = 0
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
                    line1.image(image.numpy(), width=250, caption='original image')
                except Exception:
                    continue
                x = x.permute(0, 3, 1, 2)
                y = F.interpolate(x, size=(128, 128))
                x = F.interpolate(x, size=(64, 64))
                x = F.interpolate(x, size=(128, 128))
                line2.image(x.permute(0, 2, 3, 1).detach().numpy(), width=250, caption='Downsampled')
                prog_bar.progress(i / len(dataset))
                i += 1
                out = net(x.detach().cuda().float())
                diff = torch.abs(out.detach().cpu() - y.detach().cpu())
                diff_image = diff.permute(0, 2, 3, 1).numpy()
                line5.image(diff_image, width=250, caption='absolute difference')
                line3.image(out.permute(0, 2, 3, 1).detach().cpu().numpy(), width=250, caption='Reconstructed')
                loss = 1 / criterion(out, y.detach().cuda().float())
                # loss = criterion(out, y.detach().cuda().float())

                line4.write(f'LOSS: {loss.detach().cpu()}')
                # loss.backward()
                # optimizer.step()

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
        #
        # # --------------------
        #
        #
        # x = x.permute(0, 3, 1, 2)
        # y = 1 * x

    # st.write('TESTING/EVALUATING')
    #     i = 1
    #     with torch.no_grad():
    #         for batch in train_loader:
    #             for sample in batch:
    #                 x,y = sample
    #                 test_progress_bar.progress(i / len(dataset))
    #                 i += 1
    #                 out = net(x.cuda().float())


if __name__ == '__main__':
    run_app()
