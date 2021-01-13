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
from torchvision import datasets

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
            Input is a image tensor, where channel is last.
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


DATASET_ROOT_DIR = 'C:/Users/Admin/Dataset'
train_dataset = datasets.CIFAR100(root=DATASET_ROOT_DIR, train=True, download=True)

sample = train_dataset[0]
image, target = sample

st.image(image, caption=f'Sample Image, class:\t{target}')
st.write(f'image type:\t{type(image)}  \n\ttarget type: {type(target)}')
