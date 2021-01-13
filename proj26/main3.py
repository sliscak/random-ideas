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
from torchvision import datasets, transforms

def preprocess(image, image_size=(64, 64)):
    """"
        Resizes, normalizes the image and returns it as numpy array.
    """
    image = image.resize(image_size)
    image = np.array(image) / 255
    return image


class NeuralMem(nn.Module):
    def __init__(self, image_size=(64, 64), index_pretrain=False, kernel_size=(32, 32)):
        super(NeuralMem, self).__init__()
        # res = faiss.StandardGpuResources()
        # self.mem = faiss.IndexFlatL2(25) # size of one tile/kernel
        # self.mem = faiss.index_cpu_to_gpu(res, 0, self.mem)
        self.output_size = image_size
        self.kernel = kernel_size
        self.dimensions = int(np.product(self.kernel) * self.output_size[2])
        self.stride = 4
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
            Input is a image tensor, where channel is last.
        """
        output_size = image.shape

        image = image.permute(2, 0, 1)
        image = image.unsqueeze(0)
        unfolded = torch.nn.functional.unfold(image, kernel_size=self.kernel, stride=self.stride, padding=self.padding)
        unfolded = unfolded.squeeze(0)
        unfolded = unfolded.permute(1, 0)
        with st.spinner('INFERENCE in progress...'):
            # progress_bar = st.progress(0)
            unfolded = unfolded.contiguous().numpy().astype('float32')
            ds, ks = self.mem.search(unfolded, 1)
            out_classes = Counter()
            for i, mappings_id in enumerate(ks):
                mappings_id = int(mappings_id[0])
                # if len(self.pattern_mappings[mappings_id].most_common(10)) > 2:
                #     st.write(f'MOST COMMON: {self.pattern_mappings[mappings_id].most_common(3)}')
                # out_classes = self.pattern_mappings[mappings_id].most_common(1)[0][0]
                out_classes.update([c[0] for c in self.pattern_mappings[mappings_id].most_common()])
                # progress_bar.progress(i / (unfolded.shape[0] - 1))
        # output_class = out_classes.most_common(1)[0][0]
        # return output_class
        output_classes = [c[0] for c in out_classes.most_common(3)]
        return output_classes

    def add(self, input_example, target:int):
        # takes one image and target as input
        # input shape of the example image is HxWxC and is changed into CxHxW
        image = input_example.permute(2, 0, 1)
        image = image.unsqueeze(0)

        unfolded = torch.nn.functional.unfold(image, kernel_size=self.kernel, stride=self.stride, padding=self.padding)
        unfolded = unfolded.squeeze(0)
        unfolded = unfolded.permute(1, 0)
        unfolded = unfolded.contiguous().numpy().astype('float32')

        if self.index_pretrain:
            if not self.mem.is_trained:
                self.mem.train(unfolded)
        # Make sure the resolution is the same or the loop is gonna get wrong!
        # Maybe use numpy.split() method on unfolded
        # TODO: make sure the indexing is correct!
        for i, pattern in enumerate(unfolded):
            pattern = pattern.reshape(1, -1)

            # d is distance, k is key
            d, k = self.mem.search(pattern, 1)

            k = int(k[0][0])

            # if the pattern is not in self.mem add it.
            if d[0][0] > 0:
                self.mem.add(pattern)
                # or use search which is slower to get the key
                k = self.mem.ntotal - 1

            mappings = self.pattern_mappings.get(k)
            if mappings is None:
                mappings = Counter([target])
                self.pattern_mappings[k] = mappings
            else:
                self.pattern_mappings[k].update([target])

image_sizes = [(2**x, 2**x, 3) for x in range(3, 10)]
TRAINING_IMAGE_SIZE = st.sidebar.selectbox(
    'Choose TRAINING image size', options=image_sizes, index=1)

INPUT_IMAGE_SIZE = st.sidebar.selectbox(
    'Choose INPUT image size', options=image_sizes, index=1)

add_selectbox = st.sidebar.selectbox(
    "Use index pretrain?",
    ("YES", "NO"), index=0
)
INDEX_PRETRAIN = True if add_selectbox == "YES" else False

kernel_sizes = [(x,x) for x in range(1, 33)]
KERNEL_SIZE = st.sidebar.selectbox(
    'Choose kernel size', options=kernel_sizes, index=4)

# col1_1, col1_2 = st.beta_columns(2)
# input_ph = st.empty() #button_col -> transform button
# train_int_col, train_out_col= st.beta_columns(2)
# input_col, output_col = st.beta_columns(2)
# rand_input_col, rand_output_col = st.beta_columns(2)
# progress_ph = st.empty()
# train_status1, train_status2 = st.beta_columns(2)
train_image_ph = st.empty()
train_progress_ph = st.empty()
evaluation_image_ph = st.empty()
evaluation_progress_ph = st.empty()

evaluation_status_ph = st.empty()


net = NeuralMem(image_size=TRAINING_IMAGE_SIZE, index_pretrain=INDEX_PRETRAIN, kernel_size=KERNEL_SIZE)

DATASET_ROOT_DIR = 'C:/Users/Admin/Dataset'
train_dataset = datasets.CIFAR100(root=DATASET_ROOT_DIR, train=True, download=True)
test_dataset = datasets.CIFAR100(root=DATASET_ROOT_DIR, train=True, download=True) # set to true for now

# Training
train_progress_ph.progress(0)
with st.spinner('Training...'):
    for i, sample in enumerate(train_dataset):
        if i == 1050:
            st.success('Training complete!')
            break
        image, target = sample
        image = preprocess(image, image_size=TRAINING_IMAGE_SIZE[0:2])
        train_image_ph.image(image, caption=f'training image, class:\t{target}', width=250)
        image = torch.tensor(image)
        net.add(input_example=image, target=target)
        train_progress_ph.progress(i / (1050 - 1))
    # train_progress_ph.progress(i / (len(train_dataset) - 1))

evaluation_progress_ph.progress(0)
transformation = transforms.Compose([
    transforms.CenterCrop(10),
    # transforms.ColorJitter(),
    # transforms.RandomAffine(15),
    # transforms.ToTensor(),
])
with st.spinner('Evaluating...'):
    for i, sample in enumerate(test_dataset):
        if i == 50:
            st.success('Evaluation complete!')
            break
        image, target = sample
        image = preprocess(image, image_size=TRAINING_IMAGE_SIZE[0:2])
        evaluation_image_ph.image(image, caption=f'evaluation image, class:\t{target}', width=250)
        image = torch.tensor(image)
        image = image.permute(2, 0, 1)
        image = transformation(image)
        image = image.permute(1, 2, 0)
        # image = image + (torch.rand_like(image)* 0.05)
        out = net(image)
        st.write(f'GroundTruth | Output: {target} | {out}')
        # evaluation_status_ph.write(f'OUT: {out}')
        evaluation_progress_ph.progress(i / (50 - 1))

# sample = train_dataset[0]
# image, target = sample
#
# st.image(image, caption=f'Sample Image, class:\t{target}')
# st.write(f'image type:\t{type(image)}  \n\ttarget type: {type(target)}')
