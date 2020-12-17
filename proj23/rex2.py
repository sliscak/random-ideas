import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import os
# import kornia
from PIL import Image
from time import sleep
from collections import deque
from new_datasets import WikiDataset
from collections import Counter, namedtuple

def fun(x):
    return (x * 2) - 1

Window = namedtuple("Window", ['data', 'position'])

def make_patch_generator(image, kernel, stride=1):
    x_max, y_max, c_max = image.shape
    for x in range(0, x_max - kernel[0], stride):
        for y in range(0, y_max - kernel[1], stride):
            yield Window(
                image[x:x + kernel[0], y:y + kernel[1]],
                np.array([fun(x / x_max), fun(y / y_max), fun((x + kernel[0]) / x_max), fun((y + kernel[1]) / y_max)])
            )
            # yield image[x:x + kernel[0], y:y + kernel[1]], np.array([fun(x/x_max), fun(y/y_max), fun((x + kernel[0]) / x_max), fun((y + kernel[1]) / y_max)])

def data2text(data):
    flat = data.flatten()
    array = [chr(c) for c in flat]
    return ''.join(array)

dataset = WikiDataset(dtype=int)
cols = st.beta_columns(2)
rows1 = [col.empty() for col in cols]
# for col in cols:

for x in range(5, 20):
    patch_count = Counter()
    all_patches = {}
    for i, data in enumerate(dataset):
        # data = dataset[0]
        data = np.reshape(data, (64, 64, 1))
        generator = make_patch_generator(data, (x, 1))
        rows1[0].text(f'TEXT {i}/{len(dataset)}:\n{data2text(data)}')
        for window in generator:
            patch = window.data
            patch_str = data2text(patch)
            # patch_count.update([str(patch)])
            patch_count.update([patch_str])
            count = patch_count.get(patch_str)
            rows1[1].text(f'MAX_PATCHES: {len(all_patches)}\nCOUNT: {count}\nPATCH: {data2text(patch)}')
            all_patches[patch_str] = patch
        # if count > 6:
        #     sleep(1)
    st.write(f'EP:{x} MOST COMMON:\n{patch_count.most_common(5)}')
# print(patch_count.most_common(1)[0][1])
print('END!')