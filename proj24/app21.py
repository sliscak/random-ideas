
""""
    Learns super resolution by saving tiles from high resolution image with the pytorch unfold method
        and then similarity searches for the most similar(compared to the input image/to be scaled image) tiles
        and uses them in the fold method to generate a new image.
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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        x = image
        y = image
        sample = x, y
        return sample


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
        # self.mem = faiss.IndexFlatL2(25) # size of one tile/kernel
        # self.mem = faiss.index_cpu_to_gpu(res, 0, self.mem)
        self.output_size = image_size
        self.kernel = (15, 15)
        self.dimensions = int(np.product(self.kernel))
        self.stride = 2
        self.padding = 10
        self.patterns = []

        self.nlist = 1000
        self.quantizer = faiss.IndexFlatL2(self.dimensions)
        self.mem = index = faiss.IndexIVFFlat(self.quantizer, self.dimensions, self.nlist)

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

net = NeuralMem(image_size=(64, 64))

DATASET_PATH = st.text_input('DATASET PATH', value='C:\\Users\\Admin\\Downloads\\i\\n01514859\\')
dataset = ImageDataset(path=DATASET_PATH, size=10, image_size=(128, 128))


st_orig_image = st.empty()
st_memorized_image = st.empty()
st_loss = st.empty()

st.write(f'TRAINING:')
training_progress_bar = st.progress(0)
for i, data in enumerate(dataset):
    image_x, _ = data
    net.add(torch.tensor(image_x))
    training_progress_bar.progress(i/(len(dataset)-1))


with torch.no_grad():
    dataset = ImageDataset(path="C:\\Users\\Admin\\Dataset\\tiny-imagenet-200\\test\\images\\", size=10, image_size=(64, 64))
    st.write(f'Dataset Loaded, length: {len(dataset)}')
    for image_x, image_y in dataset:
        image_tensor = torch.tensor(image_y)
        image = image_tensor.detach().numpy()
        out = net(image_tensor)
        similarity = torch.cosine_similarity(out.flatten(), image_tensor.flatten().detach(), 0)
        out_image = out.detach().numpy()
        diff_img = np.abs(image - (out_image))
        col1, col2, col3 = st.beta_columns(3)
        col1.image(image, caption='Ground Truth image', width=250)
        col2.image(out_image, caption=f'image from memory| similarity: {similarity.detach()}', width=250)
        col3.image(diff_img, caption=f'diff image', width=250)

    # dataset = ImageDataset(path=DATASET_PATH, size=20)
    # for image_x, image_y in dataset:
    #     image_tensor = torch.tensor(image_y)
    #     image = image_tensor.detach().numpy()
    #     out = net(image_tensor)
    #     similarity = torch.cosine_similarity(out.flatten(), image_tensor.flatten().detach(), 0)
    #     out_image = out.detach().numpy()
    #     diff_img = np.abs(image - (out_image))
    #     col1, col2, col3 = st.beta_columns(3)
    #     col1.image(image, caption='Ground Truth image', width=250)
    #     col2.image(out_image, caption=f'image from memory| similarity: {similarity.detach()}', width=250)
    #     col3.image(diff_img, caption=f'diff image', width=250)

    # dataset = ImageDataset(path="C:\\Users\\Admin\\Dataset\\tiny-imagenet-200\\test\\images\\", size=10)
    # for image_x, image_y in dataset:
    #     image_tensor = torch.tensor(image_y)
    #     image = image_tensor.detach().numpy()
    #     out = net(image_tensor)
    #     similarity = torch.cosine_similarity(out.flatten(), image_tensor.flatten().detach(), 0)
    #     out_image = out.detach().numpy()
    #     diff_img = np.abs(image - out_image)
    #     col1, col2, col3 = st.beta_columns(3)
    #     col1.image(image, caption='Ground Truth image', width=250)
    #     col2.image(out_image, caption=f'image from memory| similarity: {similarity.detach()}', width=250)
    #     col3.image(diff_img, caption=f'diff image', width=250)
