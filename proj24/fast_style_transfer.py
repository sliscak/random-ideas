
""""
    Fast StyleTransfer
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

def load_img(path: str = "image.png", image_size=(64, 64), gray_scale=False):
    """"
        Loads an image, resizes it and returns it as numpy array.
    """
    image = Image.open(path)
    image = image.resize(image_size)
    if gray_scale:
        image = ImageOps.grayscale(image)
        # normalize
        image = np.array(image) / 255
        image = np.expand_dims(image, axis=2)
    else:
        # normalize
        image = np.array(image) / 255
        if len(image.shape) < 3:
            return None
    return image

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
                    if image is None:
                        continue
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


class NeuralMem(nn.Module):
    def __init__(self, image_size=(64, 64)):
        super(NeuralMem, self).__init__()
        # res = faiss.StandardGpuResources()
        # self.mem = faiss.IndexFlatL2(25) # size of one tile/kernel
        # self.mem = faiss.index_cpu_to_gpu(res, 0, self.mem)
        self.output_size = image_size
        self.kernel = (5, 5)
        self.dimensions = int(np.product(self.kernel) * self.output_size[2])
        self.stride = 1
        self.padding = 10

        self.nlist = 100
        # self.mem = faiss.IndexFlatL2(self.dimensions)
        self.quantizer = faiss.IndexFlatL2(self.dimensions)
        self.mem = index = faiss.IndexIVFFlat(self.quantizer, self.dimensions, self.nlist)

    def forward(self, image):
        """"
            Input is a image tensor
        """
        st.write(image.shape)
        image = image.permute(2, 0, 1)
        image = image.unsqueeze(0)
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
                                       output_size=self.output_size[0:2],
                                       kernel_size=self.kernel,
                                       stride=self.stride,
                                       padding=self.padding)
        # return out.squeeze(0).squeeze(0)/out.squeeze(0).squeeze(0).max(0)
        # out = torch.nn.functional.interpolate(out, self.output_size)
        out = out.squeeze(0).squeeze(0) / out.flatten().max()
        out = out.permute(1,2,0)
        st.write(f'Out shape: {out.shape}')
        return out

    def add(self, image):
        # takes tensor array as input image.
        # input shape is HxWxC and is changed into CxHxW
        st.write(image.shape)
        image = image.permute(2, 0, 1)
        image = image.unsqueeze(0)
        unfolded = torch.nn.functional.unfold(image, kernel_size=self.kernel, stride=self.stride, padding=self.padding)
        st.write(unfolded.shape)
        unfolded = unfolded.squeeze(0)
        unfolded = unfolded.permute(1, 0)
        patterns = None
        for i, pattern in enumerate(unfolded):
            # st.write(pattern.shape)
            # if (i%2) != 0:
            pattern = pattern.unsqueeze(0)
            pattern = pattern.numpy().astype('float32')
            if patterns is None:
                patterns = pattern
            else:
                patterns = np.concatenate((patterns, pattern))
        if not self.mem.is_trained:
            self.mem.train(patterns)
        self.mem.add(patterns)

# IMAGE_SIZE
net = NeuralMem(image_size=(64, 64, 3))

TRAIN_IMAGE_PATH = st.text_input('IMAGE PATH', value='img.jpg')
train_image = load_img(TRAIN_IMAGE_PATH, image_size=(64, 64), gray_scale=False)

st.image(train_image, width=250, caption='training image')

st_orig_image = st.empty()
st_memorized_image = st.empty()
st_loss = st.empty()

st.write(f'TRAINING:')
net.add(torch.tensor(train_image))
st.write(f'TRAINED!')

IMAGE_PATH = st.text_input('IMAGE PATH', value='IMAGE.jpg')
image = load_img(IMAGE_PATH, image_size=(64, 64), gray_scale=False)
st.image(image, width=250, caption='input image')
output = net(torch.tensor(image)).numpy()
st.image(output, width=250, caption='output image')