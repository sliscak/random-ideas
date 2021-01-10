""""
    Fast and Robust Image StyleTransfer and Colorization by providing INPUT and OUTPUT example pairs and using similarity search.
    DONE: remove/skip duplicate patterns/kernels from faiss index/memory
    TODO: learn/train at lower resolution
    TODO: rotate and mirror the patterns/kernels and use other transformations and augmentations.
    TODO: increase speed by parallelizing the pattern retrieval(similarity search)
    TODO: add a small cache for recently found(retrieved) patterns.

"""

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
        image = np.tile(image, (1, 3))
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
        gray_image = np.tile(gray_image, (1, 3))

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


class ImageDataset(Dataset):
    def __init__(self, path='', image_size=(64, 64), size=None, testing=False):
        self.image_folder_path = path
        self.data = []

        # TODO: Load on the fly
        for root, dirs, files in os.walk(path, topdown=False):
            for image_name in files:
                if '.JPEG' in image_name:
                    image_path = self.image_folder_path + image_name
                    pair = load_img(image_path, image_size=image_size, return_pair=True)
                    if pair is None:
                        continue
                    else:
                        gray_image, image = pair
                    self.data.append((gray_image, image))
                if size is not None:
                    if len(self.data) >= size:
                        break

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, gray_image = self.data[idx]
        x = image
        y = gray_image
        sample = x, y
        return sample

class NeuralMem(nn.Module):
    def __init__(self, image_size=(64, 64), index_pretrain=False, kernel_size=(32, 32)):
        super(NeuralMem, self).__init__()
        # res = faiss.StandardGpuResources()
        # self.mem = faiss.IndexFlatL2(25) # size of one tile/kernel
        # self.mem = faiss.index_cpu_to_gpu(res, 0, self.mem)
        self.output_size = image_size
        self.kernel = kernel_size
        self.dimensions = int(np.product(self.kernel) * self.output_size[2])
        self.stride = 1
        self.padding = 10
        self.pattern_mappings = {}
        self.index_pretrain = index_pretrain

        self.nlist = 100
        # self.mem = faiss.IndexFlatL2(self.dimensions)
        if self.index_pretrain:
            self.quantizer = faiss.IndexFlatL2(self.dimensions)
            self.mem = faiss.IndexIVFFlat(self.quantizer, self.dimensions, self.nlist)

        else:
            # self.quantizer = faiss.IndexFlatL2(self.dimensions)
            # self.mem = faiss.IndexIVFFlat(self.quantizer, self.dimensions, self.nlist)
            self.mem = faiss.IndexFlatL2(self.dimensions)

        self.mem2 = faiss.IndexFlatL2(self.dimensions)


    def forward(self, image):
        """"
            Input is a image tensor
        """
        output_size = image.shape

        # st.write(image.shape)
        image = image.permute(2, 0, 1)
        image = image.unsqueeze(0)
        unfolded = torch.nn.functional.unfold(image, kernel_size=self.kernel, stride=self.stride, padding=self.padding)
        unfolded = unfolded.squeeze(0)
        unfolded = unfolded.permute(1, 0)
        with st.spinner('INFERENCE in progress...'):
            progress_bar = st.progress(0)
            unfolded = unfolded.contiguous().numpy().astype('float32')
            ds, ks = self.mem.search(unfolded, 1)
            out = None
            for i, mappings_id in enumerate(ks):
                mappings_id = int(mappings_id[0])
                # if len(self.pattern_mappings[mappings_id].most_common(10)) > 2:
                #     st.write(f'MOST COMMON: {self.pattern_mappings[mappings_id].most_common(3)}')
                pattern_id = self.pattern_mappings[mappings_id].most_common(1)[0][0]
                reconstructed = self.mem2.reconstruct(pattern_id)
                found = torch.tensor(reconstructed).unsqueeze(0)
                if out is None:
                    out = found
                else:
                    out = torch.cat((out, found), 0)
                progress_bar.progress(i / (unfolded.shape[0] - 1))
        out = out.permute(1, 0)
        out = out.unsqueeze(0)
        out = torch.nn.functional.fold(out,
                                       output_size=output_size[0:2],
                                       kernel_size=self.kernel,
                                       stride=self.stride,
                                       padding=self.padding)
        out = out.squeeze(0).squeeze(0) / out.flatten().max()
        out = out.permute(1, 2, 0)
        # st.write(f'Out shape: {out.shape}')
        return out

    def add(self, input_example, output_example, progress_ph):
        # train_status_ph....
        # takes two tensor arrays as input.
        # input shape of each example is HxWxC and is changed into CxHxW
        # both examples need to have the same resolution
        t0 = time()
        image1 = input_example.permute(2, 0, 1)
        image2 = output_example.permute(2, 0, 1)

        image1 = image1.unsqueeze(0)
        image2 = image2.unsqueeze(0)
        unfolded1 = torch.nn.functional.unfold(image1, kernel_size=self.kernel, stride=self.stride, padding=self.padding)
        unfolded2 = torch.nn.functional.unfold(image2, kernel_size=self.kernel, stride=self.stride, padding=self.padding)

        unfolded1 = unfolded1.squeeze(0)
        unfolded2 = unfolded2.squeeze(0)

        unfolded1 = unfolded1.permute(1, 0)
        unfolded2 = unfolded2.permute(1, 0)

        # with st.spinner('TRAINING in progress...'):
        old_ntotal = self.mem.ntotal
        unfolded1 = unfolded1.contiguous().numpy().astype('float32')
        unfolded2 = unfolded2.contiguous().numpy().astype('float32')
        if self.index_pretrain:
            if not self.mem.is_trained:
                self.mem.train(unfolded1)
        # Make sure the resolution is the same or the loop is gonna get wrong!
        # Maybe use numpy.split() method on unfolded1
        # TODO: make sure the indexing is correct!
        train_progress_bar = progress_ph.progress(0)
        for i, pattern1 in enumerate(unfolded1):
            pattern1 = pattern1.reshape(1, -1)
            pattern2 = unfolded2[i].reshape(1, -1)
            # pattern1 = pattern1.unsqueeze(0)
            # pattern2 = unfolded2[i].unsqueeze(0)

            # pattern1 = pattern1.numpy().astype('float32')
            # pattern2 = pattern2.numpy().astype('float32')

            d1, k1 = self.mem.search(pattern1, 1)
            d2, k2 = self.mem2.search(pattern2, 1)

            k1 = int(k1[0][0])
            k2 = int(k2[0][0])

            # if the pattern1 is not in self.mem add it.
            if d1[0][0] > 0:
                self.mem.add(pattern1)
                k1 = self.mem.ntotal - 1
                # if the pattern2 is not in self.mem2 add it.
            if d2[0][0] > 0:
                self.mem2.add(pattern2)
                k2 = self.mem2.ntotal - 1

            mappings = self.pattern_mappings.get(k1)
            if mappings is None:
                mappings = Counter([k2])
                self.pattern_mappings[k1] = mappings
            else:
                self.pattern_mappings[k1].update([k2])
                # mappings.update([k2])
            train_progress_bar.progress(i / (len(unfolded1) - 1))
            # st.success(f'LEARNED: {self.mem.ntotal-old_ntotal}\tpatterns in {time() -  t0} seconds!')


image_sizes = [(2**x, 2**x, 3) for x in range(5, 10)]
TRAINING_IMAGE_SIZE = st.sidebar.selectbox(
    'Choose TRAINING image size', options=image_sizes, index=1)
OUTPUT_IMAGE_SIZE = st.sidebar.selectbox(
    'Choose OUTPUT image size', options=image_sizes, index=1)
# IMAGE_SIZE = (64, 64, 3)
# IMAGE_SIZE = (128, 128, 3)
add_selectbox = st.sidebar.selectbox(
    "Use index pretrain?",
    ("YES", "NO"), index=0
)

kernel_sizes = [(x,x) for x in range(1, 33)]
KERNEL_SIZE = st.sidebar.selectbox(
    'Choose kernel size', options=kernel_sizes, index=4)
use_dataset = st.sidebar.checkbox('Use dataset?', True)
INDEX_PRETRAIN = True if add_selectbox == "YES" else False
net = NeuralMem(image_size=TRAINING_IMAGE_SIZE, index_pretrain=INDEX_PRETRAIN, kernel_size=KERNEL_SIZE)

with st.beta_expander("FAST AND ROBUST IMAGE STYLETRANSFER AND COLORIZATION", expanded=True):
    # header1 = st.write('## FAST AND ROBUST IMAGE STYLETRANSFER AND COLORIZATION')
    header2 = st.markdown('#### by providing input and output example image pairs and by using similarity search')
    header3 = st.markdown('##### Transfer the style of images by providing input and output example images.')
    header4 = st.markdown('##### Colorize images by providing black-white or grayscale input and colored output example images(like grayscale photo as input example and colored photo as output example for training)')

# video_file = open('tutorial.webm', 'rb')
# video_bytes = video_file.read()
# st.video(video_bytes)


col1_1, col1_2 = st.beta_columns(2)
input_ph = st.empty() #button_col -> transform button
train_int_col, train_out_col= st.beta_columns(2)
input_col, output_col = st.beta_columns(2)
rand_input_col, rand_output_col = st.beta_columns(2)
progress_ph = st.empty()
train_status1, train_status2 = st.beta_columns(2)


if not use_dataset:
    uploaded_inp_example = col1_1.file_uploader("Choose INPUT EXAMPLE for training", type=['png', 'jpg'])
    uploaded_out_example = col1_2.file_uploader("Choose OUTPUT EXAMPLE for training", type=['png', 'jpg'])
    uploaded_file = input_ph.file_uploader("Choose input image", type=['png', 'jpg'])

    if uploaded_inp_example is not None and uploaded_out_example is not None and uploaded_file is not None:
        train_inp_example = preprocess(uploaded_inp_example, image_size=TRAINING_IMAGE_SIZE[0:2], gray_scale=False)
        train_int_col.image(train_inp_example, caption="INPUT EXAMPLE", width=250)
        train_inp_example = torch.tensor(train_inp_example)

        train_out_example = preprocess(uploaded_out_example, image_size=TRAINING_IMAGE_SIZE[0:2], gray_scale=False)
        train_out_col.image(train_out_example, caption="OUTPUT EXAMPLE", width=250)
        train_out_example = torch.tensor(train_out_example)

        # image = preprocess(uploaded_file, image_size=IMAGE_SIZE[0:2], gray_scale=False)
        image = preprocess(uploaded_file, image_size=OUTPUT_IMAGE_SIZE[0:2], gray_scale=False)
        input_col.image(image, width=250, caption='input image')

        net.add(train_inp_example, train_out_example)
        output = net(torch.tensor(image)).numpy()
        output_col.image(output, width=250, caption='output image')
else:
    uploaded_file = input_ph.file_uploader("Choose input image", type=['png', 'jpg'])

    if uploaded_file is not None:
        DATASET_PATH = 'C:/Users/Admin/Dataset/mini-dataset/test/images/'
        dataset = ImageDataset(path=DATASET_PATH, image_size=TRAINING_IMAGE_SIZE[0:2], size=100)
        # image = preprocess(uploaded_file, image_size=IMAGE_SIZE[0:2], gray_scale=False)
        image = preprocess(uploaded_file, image_size=OUTPUT_IMAGE_SIZE[0:2], gray_scale=True)
        input_col.image(image, width=250, caption='input image')

        dataset_train_progress_bar = st.progress(0)
        for i, example in enumerate(dataset):
            train_inp_example, train_out_example = example
            # col1_1.image(train_inp_example, width=250)
            # col1_2.image(train_out_example, width=250)
            train_inp_example = torch.tensor(train_inp_example)
            train_out_example = torch.tensor(train_out_example)
            # print(f'INP: {train_inp_example.shape}')
            # print(f'OUT: {train_out_example.shape}')
            net.add(train_inp_example, train_out_example, progress_ph)
            dataset_train_progress_bar.progress((i+1)/(len(dataset)))
        output = net(torch.tensor(image)).numpy()
        output_col.image(output, width=250, caption='output image')



#
# image = torch.rand(IMAGE_SIZE)
# rand_input_col.image(image.numpy(), width=250, caption='random input image')
# output = net(torch.tensor(image)).numpy()
# rand_output_col.image(output, width=250, caption='output image')
