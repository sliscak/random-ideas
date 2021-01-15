
import os

import faiss
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
from collections import Counter
from PIL import Image
from PIL import ImageOps
from torch.utils.data import Dataset
from torch import optim
from time import sleep

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


def load_img(path: str = "image.png", image_size=(64, 64), gray_scale=False, return_pair=False):
    """"
        Loads an image, resizes it and returns it as numpy array.
        Options include: output image as grayscale and output grayscale and colored image pair
    """
    image = Image.open(path)
    image = image.resize(image_size)
    if return_pair:
        gray_image = ImageOps.grayscale(image)
        gray_image = np.array(gray_image) / 255
        gray_image = np.expand_dims(gray_image, axis=2)

        image = np.array(image) / 255
        if len(image.shape) < 3:
            return None
        else:
            return (gray_image, image)
    else:
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

# models is called OnePatch
class PatchLayer(nn.Module):
    def __init__(self, input_size=(64, 64, 3), kernel_size=(32, 32), num_patches: int = 100):
        super(PatchLayer, self).__init__()
        self.output_size = input_size
        self.kernel = kernel_size
        self.dimensions = int(np.product(self.kernel) * self.output_size[2])
        self.stride = 1
        self.padding = 10
        self.patches = nn.ParameterList([nn.Parameter(torch.rand(1,self.dimensions)) for i in range(num_patches)])

    def forward(self, image):
        """"
            Input is a image tensor
        """
        output_size = image.shape
        image = image.permute(2, 0, 1)
        image = image.unsqueeze(0)
        unfolded = torch.nn.functional.unfold(image, kernel_size=self.kernel, stride=self.stride, padding=self.padding)
        unfolded = unfolded.squeeze(0)
        unfolded = unfolded.permute(1, 0)
        output = None
        for patch in self.patches:
            cosim = 1 - torch.cosine_similarity(unfolded, patch)
            cosim = cosim.sum(0) / unfolded.shape[0] # normalize the similarity
            cosim = cosim.unsqueeze(0)#.unsqueeze(0)
            # cosim = cosim.reshape(1)
            if output is None:
                output = cosim
            else:
                output = torch.cat((output, cosim))
        return output

class PatchNet(nn.Module):
    def __init__(self, input_size=(64, 64, 3), kernel_size=(32, 32)):
        super(PatchNet, self).__init__()
        # self.block = nn.ModuleList(
        #     PatchLayer
        # )
        self.first_layer = PatchLayer(input_size=input_size, kernel_size=kernel_size, num_patches=100)
        self.last_layer = PatchLayer(input_size=(10, 10, 1), kernel_size=(2, 2), num_patches=1)

    def forward(self, image):
        """"
            Input is a image tensor
        """
        o = self.first_layer(image) # o is of size 100
        o = o.reshape(10,10,1) # use BxWxH --> 10x10x1   because 10x10 == 100
        o = self.last_layer(o)
        # st.write(o.shape)
        return o


image_sizes = [(2**x, 2**x, 3) for x in range(5, 10)]
TRAINING_IMAGE_SIZE = st.sidebar.selectbox(
    'Choose TRAINING image size', options=image_sizes, index=1)

kernel_sizes = [(x,x) for x in range(1, 33)]
KERNEL_SIZE = st.sidebar.selectbox(
    'Choose kernel size', options=kernel_sizes, index=4)

stride_vals = [x for x in range(1, 11)]
STRIDE = st.sidebar.selectbox(
    'Choose stride value', options=stride_vals, index=4)

net = PatchNet(input_size=TRAINING_IMAGE_SIZE, kernel_size=KERNEL_SIZE)
optimizer = optim.AdamW(params=net.parameters(), lr=0.03)

# with st.beta_expander("FAST AND ROBUST IMAGE STYLETRANSFER AND COLORIZATION", expanded=True):
#     # header1 = st.write('## FAST AND ROBUST IMAGE STREETCARS AND COLORIZATION')
#     header2 = st.markdown('#### by providing input and output example image pairs and by using similarity search')
#     header3 = st.markdown('##### Transfer the style of images by providing input and output example images.')
#     header4 = st.markdown('##### Colorize images by providing black-white or grayscale input and colored output example images(like grayscale photo as input example and colored photo as output example for training)')

# video_file = open('tutorial.webm', 'rb')
# video_bytes = video_file.read()
# st.video(video_bytes)

with st.beta_expander("LEARN A SINGLE PATTERN FROM ONE IMAGE OR MULTIPLE IMAGES THAT WILL REPRESENT THAT PARTICULAR IMAGE OR IMAGE CLASS", expanded=True):
    pass

col1_1, col1_2 = st.beta_columns(2)
input_ph = st.empty()
train_int_col, train_out_col= st.beta_columns(2)
input_col, output_col = st.beta_columns(2)
output_col = output_col.empty()
rand_input_col, rand_output_col = st.beta_columns(2)
loss_ph = st.empty()

uploaded_file = input_ph.file_uploader("Choose input image", type=['png', 'jpg']    )

if uploaded_file is not None:
    with st.spinner('TRAINING in progress...'):
        image = preprocess(uploaded_file, image_size=TRAINING_IMAGE_SIZE[0:2], gray_scale=False)
        # image has shape HxWxC
        input_col.image(image, width=250, caption='input image')
        while True:
            optimizer.zero_grad()
            loss = net(torch.tensor(image))
            loss.backward()
            optimizer.step()
            loss_ph.write(f'LOSS: {loss.clone().detach().numpy()}')
            # patch = net.patch.clone().detach()
            # patch = net.layer.patches[0].clone().detach()
            # patch = patch.reshape(KERNEL_SIZE[0], KERNEL_SIZE[1], 3).numpy()
            # # output = net(torch.tensor(image)).numpy()
            # output_col.image(patch, width=250, caption='output image')
            sleep(1)


#
# image = torch.rand(IMAGE_SIZE)
# rand_input_col.image(image.numpy(), width=250, caption='random input image')
# output = net(torch.tensor(image)).numpy()
# rand_output_col.image(output, width=250, caption='output image')
