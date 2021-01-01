""""
    Learns a single (super) pattern from all images in the dataset, (the batch has the size of the whole dataset)
    Here the super pattern represents one image class.

    The interesting thing is that the learned super pattern and the image average pattern(made from averaging all images in dataset)
        look nearly the same. Gradient descent/Backpropagation is slow and the averaging operation is faster.
"""

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
    def __init__(self, path='', size=None, testing=False):
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
                if size is not None:
                    if len(self.data) >= size:
                        break
                    # break
                    # st.write('WORKING!')
        # if not testing:
        #     self.data = self.data[0:10]
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
        self.patterns = []
        self.key_classes = []

    def forward(self, x):
        similarities = []
        similarity_values = []
        patterns = None
        for idx, pattern in enumerate(self.patterns):
            similarity = torch.cosine_similarity(x.flatten(), pattern.flatten(), 0)
            similarities.append((similarity, idx))
            similarity_values.append(similarity)
            # patterns.append(pattern.flatten() * similarity)
            # if patterns is None:
            #     patterns = pattern.flatten() * similarity
            # else:
            #     patterns += pattern.flatten() * similarity
        similarity_values = torch.softmax(torch.tensor(similarity_values), 0)
        patterns = None
        print(similarity_values)
        for idx, pattern in enumerate(self.patterns):
            if patterns is None:
                patterns = pattern.flatten() * similarity_values[idx]
            else:
                patterns += pattern.flatten() * similarity_values[idx]
        patterns = patterns / len(self.patterns)
        sum = patterns.reshape(128, 128)
        # sum = sum.clamp(0, 1)
        # similarity_values = torch.tensor(similarity_values)
        # similarity_avg = torch.mean(similarity_values, 0)
        # # print(similarity_avg)
        # similarities.sort(key=lambda l: l[0],reverse=True)
        # most_similar_idx = similarities[0][1]
        # most_similar_pattern = self.patterns[most_similar_idx]
        # st.write(f'CLASS SIMILARITY AVERAGE: {similarity_avg.detach()}')
        return sum

    def add(self, key, key_class):
        self.patterns.append(key)
        self.key_classes.append(key_class)

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.patterns = []
#         self.key_classes = []
#
#     def forward(self, x):
#         similarities = []
#         idxs = []
#         patts = []
#         for pattern, idx in self.patterns:
#             similarity = torch.cosine_similarity(x.flatten(), pattern.flatten(), 0)
#             similarities.append(similarity)
#             idxs.append(idx)
#             patts.append((similarity, idx))
#         patts.sort(key=lambda l: l[0],reverse=True)
#         most_similar_idx = patts[0][1]
#         most_similar_pattern = self.patterns[most_similar_idx][0]
#         st.write(similarities)
#         return most_similar_pattern
#
#     def add(self, key, key_class):
#         self.patterns.append([key, key_class])
#         self.key_classes.append(key_class)

net = Net()
# optimizer = optim.AdamW(net.parameters(), lr=0.005)
# criterion = nn.MSELoss()

DATASET_PATH = st.text_input('DATASET PATH', value='C:\\Users\\Admin\\Downloads\\i\\n01514859\\')
dataset = ImageDataset(path=DATASET_PATH, size=20)

# for image_x, image_y in dataset:
#     st.image(image_x)

# st.stop()
st_orig_image = st.empty()
st_memorized_image = st.empty()
st_loss = st.empty()

for image_x, image_y in dataset:
    image_tensor = torch.tensor(image_y)
    net.add(torch.tensor(image_x), key_class=0)

print(net(torch.tensor(dataset[2][1])))
# exit()
average = torch.tensor([0])
# images = []
for image_x, image_y in dataset:
    average += image_y
    # images.append(image_y.flatten())
average /= len(dataset)
# average = np.median(images, 0)
# average = torch.tensor([average]).reshape(128, 128)

with torch.no_grad():
    dataset = ImageDataset(path=DATASET_PATH, size=50)
    for image_x, image_y in dataset:
        image_tensor = torch.tensor(image_y)
        image = image_tensor.detach().numpy()
        out = net(image_tensor)
        # loss = criterion(out.flatten(), image_tensor.flatten().detach())
        loss = 1 - torch.cosine_similarity(out.flatten(), image_tensor.flatten().detach(), 0)
        # loss2 = criterion(average.flatten(), image_tensor.flatten().detach())
        loss2 = 1 - torch.cosine_similarity(average.flatten(), image_tensor.flatten().detach(), 0)
        out_image = out.detach().numpy()
        average_img = average.detach().numpy()
        diff_img = np.abs(image - (out_image))
        col1, col2, col3, col4 = st.beta_columns(4)
        col1.image(image, caption='Ground Truth image', width=200)
        col2.image(out_image, caption=f'image from memory| loss: {loss.detach()}', width=200)
        col3.image(average_img, caption=f'average image| {loss2.detach()}', width=200)
        col4.image(diff_img, caption=f'diff image', width=200)
    dataset = ImageDataset(path="C:\\Users\\Admin\\Dataset\\tiny-imagenet-200\\test\\images\\", size=10)
    for image_x, image_y in dataset:
        image_tensor = torch.tensor(image_y)
        image = image_tensor.detach().numpy()
        out = net(image_tensor)
        # loss = criterion(out.flatten(), image_tensor.flatten().detach())
        loss = 1 - torch.cosine_similarity(out.flatten(), image_tensor.flatten().detach(), 0)
        # loss2 = criterion(average.flatten(), image_tensor.flatten().detach())
        loss2 = 1 - torch.cosine_similarity(average.flatten(), image_tensor.flatten().detach(), 0)
        out_image = out.detach().numpy()
        average_img = average.detach().numpy()
        diff_img = np.abs(image - out_image)
        col1, col2, col3, col4 = st.beta_columns(4)
        col1.image(image, caption='Ground Truth image', width=200)
        col2.image(out_image, caption=f'image from memory| loss: {loss.detach()}', width=200)
        col3.image(average_img, caption=f'average image| {loss2.detach()}', width=200)
        col4.image(diff_img, caption=f'diff image', width=200)

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
