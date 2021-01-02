
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
import faiss

def load_img(path: str = "image.png", image_size=(64, 64)):
    image = Image.open(path)
    image = image.resize(image_size)
    # image = ImageOps.grayscale(image)
    # img = np.array(image)
    # normalize
    img = np.array(image)# / 255
    # img = torch.tensor(img)
    return img

class ImageDataset(Dataset):
    def __init__(self, path='', image_size=(64, 64), size=None, testing=False):
        self.image_folder_path = path
        self.data = []

        # TODO: Load on the fly
        for root, dirs, files in os.walk(path, topdown=False):
            for image_name in files:
                if '.JPEG' in image_name:
                    image_path = self.image_folder_path + image_name
                    image = load_img(image_path, image_size=image_size)
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


# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.patterns = []
#         self.key_classes = []
#
#     def forward(self, x):
#         similarities = []
#         similarity_values = []
#         patterns = None
#         for idx, pattern in enumerate(self.patterns):
#             similarity = torch.cosine_similarity(x.flatten(), pattern.flatten(), 0)
#             similarities.append((similarity, idx))
#             similarity_values.append(similarity)
#             # patterns.append(pattern.flatten() * similarity)
#             # if patterns is None:
#             #     patterns = pattern.flatten() * similarity
#             # else:
#             #     patterns += pattern.flatten() * similarity
#         similarity_values = torch.softmax(torch.tensor(similarity_values), 0)
#         patterns = None
#         print(similarity_values)
#         for idx, pattern in enumerate(self.patterns):
#             if patterns is None:
#                 patterns = pattern.flatten() * similarity_values[idx]
#             else:
#                 patterns += pattern.flatten() * similarity_values[idx]
#         patterns = patterns / len(self.patterns)
#         sum = patterns.reshape(128, 128)
#         # sum = sum.clamp(0, 1)
#         # similarity_values = torch.tensor(similarity_values)
#         # similarity_avg = torch.mean(similarity_values, 0)
#         # # print(similarity_avg)
#         # similarities.sort(key=lambda l: l[0],reverse=True)
#         # most_similar_idx = similarities[0][1]
#         # most_similar_pattern = self.patterns[most_similar_idx]
#         # st.write(f'CLASS SIMILARITY AVERAGE: {similarity_avg.detach()}')
#         return sum
#
#     def add(self, key, key_class):
#         self.patterns.append(key)
#         self.key_classes.append(key_class)

class NeuralDict(nn.Module):
    def __init__(self, classes=False):
        super(NeuralDict, self).__init__()
        self.patterns = []
        self.is_classes = classes
        if self.is_classes is True:
            self.key_classes = []

    def forward(self, x):
        # use cat function from pytorch to append tensors, instead of creating a list
        # and then use argmax and loat the most similar pattern by index.
        similarities = []
        for idx, pattern in enumerate(self.patterns):
            similarity = torch.cosine_similarity(x.flatten(), pattern.flatten(), 0)
            similarities.append((similarity, idx))
        similarities.sort(key=lambda l: l[0],reverse=True)
        most_similar_idx = similarities[0][1]
        most_similar_pattern = self.patterns[most_similar_idx]
        return most_similar_pattern

    def add(self, key, key_class=None):
        self.patterns.append(torch.tensor(key, dtype=torch.double))
        if self.is_classes is True:
            self.key_classes.append(key_class)

class NeuralMem(nn.Module):
    def __init__(self, image_size=(64, 64)):
        super(NeuralMem, self).__init__()
        # res = faiss.StandardGpuResources()
        self.nlist = 100
        self.quantizer = faiss.IndexFlatL2(225)
        self.mem = index = faiss.IndexIVFFlat(self.quantizer, 225, self.nlist)
        # self.mem = faiss.IndexFlatL2(25) # size of one tile/kernel
        # self.mem = faiss.index_cpu_to_gpu(res, 0, self.mem)
        self.output_size = image_size
        self.kernel = (15, 15)
        self.stride = 1
        self.padding = 10
        self.patterns = []

    def forward(self, image_tensor):
        """"
            Input is a image tensor
        """
        image = image_tensor.unsqueeze(0).unsqueeze(0)
        image = torch.nn.functional.interpolate(image, (128, 128))
        unfolded = torch.nn.functional.unfold(image, kernel_size=self.kernel, stride=self.stride, padding=self.padding)
        unfolded = unfolded.squeeze(0)
        unfolded = unfolded.permute(1, 0)
        out = None
        progress_bar = st.progress(0)
        for i, pattern in enumerate(unfolded):
            pattern = pattern.unsqueeze(0)
            pattern = pattern.numpy().astype('float32')
            d, k, pattern = self.mem.search_and_reconstruct(pattern, 1)
            found = torch.tensor(pattern).squeeze(0)
            if out is None:
                out = found
                # out = found.unsqueeze(0)
            else:
                out = torch.cat((out, found), 0)
                # out = torch.cat((out, found.unsqueeze(0)), 0)
            progress_bar.progress(i/unfolded.shape[0])
        out = out.permute(1, 0)
        out = out.unsqueeze(0)
        out = torch.nn.functional.fold(out,
                                       output_size=(128, 128),
                                       kernel_size=self.kernel,
                                       stride=self.stride,
                                       padding=self.padding)
        # return out.squeeze(0).squeeze(0)/out.squeeze(0).squeeze(0).max(0)
        out = torch.nn.functional.interpolate(out, self.output_size)
        out = out.squeeze(0).squeeze(0)/ out.flatten().max()
        st.write(f'Out shape: {out.shape}')
        return out

    def add(self, image_tensor):
        # needs to be of shape CxHxW or CxWxH?
        image = image_tensor.unsqueeze(0).unsqueeze(0)
        unfolded = torch.nn.functional.unfold(image, kernel_size=self.kernel, stride=self.stride, padding=self.padding)
        unfolded = unfolded.squeeze(0)
        unfolded = unfolded.permute(1, 0)
        patterns = None
        for pattern in unfolded:

            pattern = pattern.unsqueeze(0)
            pattern = pattern.numpy().astype('float32')
            if patterns is None:
                patterns = pattern
            else:
                patterns = np.concatenate((patterns, pattern))
        if not self.mem.is_trained:
            self.mem.train(patterns)
        self.mem.add(patterns)

# image_tensor = torch.rand(64,64)
# st.image(image_tensor.numpy(), width=250)
# net = NeuralMem()
# net.add(image_tensor)
# out = net(image_tensor)
# st.image(out.numpy(), width=250)
# st.write(out.shape, width=250)
# exit()


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

net = NeuralMem(image_size=(64, 64))
# optimizer = optim.AdamW(net.parameters(), lr=0.005)
# criterion = nn.MSELoss()

DATASET_PATH = st.text_input('DATASET PATH', value='C:\\Users\\Admin\\Downloads\\i\\n01514859\\')
dataset = ImageDataset(path=DATASET_PATH, size=10, image_size=(128, 128))

# for image_x, image_y in dataset:
#     st.image(image_x)

# st.stop()
# image_tensor = torch.tensor(dataset[0][0])
# image_tensor = image_tensor.unsqueeze(0)
# image_tensor = image_tensor.unsqueeze(0)
# image = image_tensor.squeeze(0).squeeze(0).numpy()
# st.image(image, 'original image')
# unfolded = torch.nn.functional.unfold(image_tensor, (5, 5), stride=5, padding=10)
# print(unfolded.shape)
# recovered_tensor = torch.nn.functional.fold(unfolded, output_size=(128, 128), kernel_size=(5, 5), stride=5, padding=10)
# image = recovered_tensor.squeeze(0).squeeze(0).clamp(0, 1).numpy()
# st.image(image, 'recovered image')
# exit()

st_orig_image = st.empty()
st_memorized_image = st.empty()
st_loss = st.empty()

for image_x, image_y in dataset:
    image_tensor = torch.tensor(image_y)
    net.add(torch.tensor(image_x))

# print(net(torch.tensor(dataset[2][1])))
# exit()
average = torch.tensor([0])
# images = []
dataset = ImageDataset(path=DATASET_PATH, size=10, image_size=(64, 64))
for image_x, image_y in dataset:
    average += image_y
    # images.append(image_y.flatten())
average /= len(dataset)
# average = np.median(images, 0)
# average = torch.tensor([average]).reshape(128, 128)

with torch.no_grad():
    dataset = ImageDataset(path=DATASET_PATH, size=20)
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
