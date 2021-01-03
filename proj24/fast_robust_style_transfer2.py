""""
    Fast and Robust StyleTransfer
"""

import os

import faiss
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from PIL import ImageOps
from torch.utils.data import Dataset


def preprocess(bytes_image, image_size=(64, 64), gray_scale=False):
    """"
        Resizes the image and returns it as numpy array.
    """
    image = Image.open(bytes_image)
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
        self.kernel = (3, 3)
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
            progress_bar.progress(i / unfolded.shape[0])
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
        out = out.permute(1, 2, 0)
        st.write(f'Out shape: {out.shape}')
        return out

    def add(self, image):
        # takes tensor array as input image.
        # input shape is HxWxC and is changed into CxHxW
        # st.write(image.shape)
        image = image.permute(2, 0, 1)
        image = image.unsqueeze(0)
        unfolded = torch.nn.functional.unfold(image, kernel_size=self.kernel, stride=self.stride, padding=self.padding)
        # st.write(unfolded.shape)
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


IMAGE_SIZE = (128, 128, 3)
net = NeuralMem(image_size=IMAGE_SIZE)

col1, col2 = st.beta_columns(2)
uploaded_file = col1.file_uploader("Choose training image")
if uploaded_file is not None:
    train_image = preprocess(uploaded_file, image_size=IMAGE_SIZE[0:2], gray_scale=False)
else:
    TRAIN_IMAGE_PATH = st.text_input('IMAGE PATH', value='img.jpg')
    train_image = load_img(TRAIN_IMAGE_PATH, image_size=IMAGE_SIZE[0:2], gray_scale=False)

train_col, input_col, output_col = st.beta_columns(3)

uploaded_file = col2.file_uploader("Choose input image")
if uploaded_file is not None:
    image = preprocess(uploaded_file, image_size=IMAGE_SIZE[0:2], gray_scale=False)
else:
    IMAGE_PATH = st.text_input('IMAGE PATH', value='image.jpg')
    image = load_img(IMAGE_PATH, image_size=IMAGE_SIZE[0:2], gray_scale=False)

train_col.image(train_image, width=250, caption='training image')
input_col.image(image, width=250, caption='input image')

# st.write(f'TRAINING:')
net.add(torch.tensor(train_image))
# st.write(f'TRAINED!')

output = net(torch.tensor(image)).numpy()
output_col.image(output, width=250, caption='output image')

rand_input_col, rand_output_col = st.beta_columns(2)
image = torch.rand(128, 128, 3)
rand_input_col.image(image.numpy(), width=250, caption='random input image')
output = net(torch.tensor(image)).numpy()
rand_output_col.image(output, width=250, caption='output image')
