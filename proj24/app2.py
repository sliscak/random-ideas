import streamlit as st
import numpy as np
from PIL import Image, ImageOps

import torch
from torch import nn
from torch import optim
from time import sleep

import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import os
from PIL import Image
from time import sleep
from collections import deque

def load_img(path: str = "image.png"):
    image = Image.open(path)
    image = image.resize((128, 128))
    # image = ImageOps.grayscale(image)
    # img = np.array(image)
    # normalize
    img = np.array(image)# / 255
    # img = torch.tensor(img)
    return img

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
                    image = Image.fromarray(image)
                    image = ImageOps.grayscale(image)
                    image = np.array(image)/255
                    self.data.append(image)
                    # break
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


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.memory = nn.Parameter(torch.rand((128, 128)))

    def forward(self, x):
        x = self.memory * x
        # x = torch.reshape(x, (30, 30))
        x = x.clamp(0, 1)
        return x


net = Net()
optimizer = optim.AdamW(net.parameters(), lr=0.005)
criterion = nn.MSELoss()

DATASET_PATH = st.text_input('DATASET PATH', value='C:\\Users\\Admin\\Downloads\\i\\n01514859\\')
dataset = ImageDataset(path=DATASET_PATH)

for image_x, image_y in dataset:
    st.image(image_x)

st.stop()
st_orig_image = st.empty()
st_memorized_image = st.empty()
st_loss = st.empty()

image_tensor = torch.rand((30, 30))
image = image_tensor.detach().numpy()
st_orig_image.image(image, caption='Ground Truth image', width=200)
image_tensors = [torch.rand((30, 30)) for i in range(10)]
# x_inp = torch.ones(1)
# out = net(x_inp)
# out_image = out.detach().numpy()
# st_memorized_image.image(out_image, caption='image from memory', width=200)
last_loss = 0
plateau = 0
# standstill = 0
while True:
    loss = torch.tensor([0.0])
    for image_tensor in image_tensors:
        image = image_tensor.detach().numpy()
        st_orig_image.image(image, caption='input image', width=200)
        optimizer.zero_grad()
        out = net(torch.ones(1))
        out_image = out.detach().numpy()
        st_memorized_image.image(out_image, caption='image from memory', width=200)
        # st.write(f'Out: {out}')
        loss += criterion(out.flatten(), image_tensor.flatten().detach())
        # loss = torch.cosine_similarity(out.flatten(), image_tensor.detach().flatten(),0)
    loss.backward()
    if loss == last_loss:
        plateau += 1
        if plateau >= 3:
            st_loss.write(f'LOSS: {loss}\tPlateau: {plateau}/{3}')
            st.write('Learning ended.')
            break
    else:
        plateau = 0
        last_loss = loss.detach()
    last_loss = loss
    optimizer.step()
    st_loss.write(f'LOSS: {loss}\tPlateau: {plateau}/{3}')
    sleep(0.05)

for image_tensor in image_tensors:
    image = image_tensor.detach().numpy()
    optimizer.zero_grad()
    out = net(torch.ones(1))
    loss = criterion(out.flatten(), image_tensor.flatten().detach())
    out_image = out.detach().numpy()
    col1, col2 = st.beta_columns(2)
    col1.image(image, caption='Ground Truth image', width=200)
    col2.image(out_image, caption=f'image from memory| loss: {loss.detach()}', width=200)
# while True:
#     for image_tensor in image_tensors:
#         out = net(torch.ones(1))
#         out_image = out.detach().numpy()
#         st_memorized_image.image(out_image, caption='image from memory', width=200)
#         # st.write(f'Out: {out}')
#         loss = criterion(out.flatten(), image_tensor.flatten().detach())
#         # loss = torch.cosine_similarity(out.flatten(), image_tensor.detach().flatten(),0)
#         loss.backward()
#         optimizer.step()
#         st_loss.write(loss)
#         sleep(0.25)
# while True:
#     out = net(torch.ones(1))
#     out_image = out.detach().numpy()
#     st_memorized_image.image(out_image, caption='image from memory', width=200)
#     # st.write(f'Out: {out}')
#     loss = criterion(out.flatten(), image_tensor.flatten().detach())
#     # loss = torch.cosine_similarity(out.flatten(), image_tensor.detach().flatten(),0)
#     loss.backward()
#     optimizer.step()
#     st_loss.write(loss)
#     sleep(0.25)

# image_tensors = [torch.rand((30, 30)) for i in range(10)]
# for image_tensor in image_tensors[0:1]:
#     # st.image(image, caption='input image', width=200)
#     x_inp = torch.ones(1)
#     out = net(x_inp)
#     out_image = out.detach().numpy()
#

# out = net(image_tensor)
# st.write(out)
