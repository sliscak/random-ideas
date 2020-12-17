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
from collections import Counter

def fun(x):
    return (x * 2) - 1

def make_patch_generator(image, shape, stride=1):
    x_max, y_max, c_max = image.shape
    for x in range(0, x_max - shape[0], stride):
        for y in range(0, y_max - shape[1], stride):
            yield image[x:x + shape[0], y:y + shape[1]], \
                  np.array([fun(x/x_max), fun(y/y_max), fun((x + shape[0])/x_max), fun((y + shape[1])/y_max)])

class ImageDataset(Dataset):
    def __init__(self, path='', testing=False):
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
                    self.data.append(image)
                    # break
                    if not testing:
                        if len(self.data) >= 30:
                            break
                    break
                    # st.write('WORKING!')
        # if not testing:
        #     self.data = self.data[0:100]
    # def load_images(self, path):

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        x = image
        y = image
        sample = x, y
        return sample


class LongConv(nn.Module):
    def __init__(self, size=1, keysize=None):
        super(LongConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=3, groups=1, kernel_size=(5, 5), stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, groups=1, kernel_size=(5, 5), stride=1, padding=2)
        self.conv3 = nn.Conv2d(in_channels=3, out_channels=3, groups=1, kernel_size=(5, 5), stride=1, padding=2)
        self.keys = nn.Parameter(torch.randn((size, keysize)))
        self.values = nn.Parameter(torch.randn((size, 3 * 3 * 5 * 5)))


    def forward(self, x):
        key = self.conv(x)
        # key = self.conv3(self.conv2(self.conv(x)))
        # key = torch.sigmoid(key)
        # key = F.interpolate(key, (32, 32))
        key = key.flatten()
        # key = torch.tanh(key)
        attention = torch.matmul(self.keys, key)
        attention = torch.softmax(attention, 0)
        attention = torch.reshape(attention, (-1, 1))

        kernel = self.values * attention
        kernel = torch.sum(kernel, 0)
        kernel = torch.reshape(kernel, (3, 3, 5, 5))

        out = torch.conv_transpose2d(x, weight=kernel, stride=3, padding=0)

        return out

class LongMem(nn.Module):
    def __init__(self, size=1, keysize=None, inch=3, outch=3,):
        super(LongMem, self).__init__()
        self.inch, self.outch = inch, outch
        self.keys = nn.Parameter(torch.randn((size, keysize)))
        self.values = nn.Parameter(torch.randn((size, inch * outch * 5 * 5)))

    def forward(self, x):
        key = 1 * x
        # key = F.interpolate(x, (32, 32))
        # key = torch.sigmoid(key)
        key = key.flatten()
        # key = torch.tanh(key)
        attention = torch.matmul(self.keys, key)
        attention = torch.softmax(attention, 0)
        attention = torch.reshape(attention, (-1, 1))

        kernel = self.values * attention
        kernel = torch.sum(kernel, 0)
        kernel = torch.reshape(kernel, (self.inch, self.outch, 5, 5))

        # out = torch.conv_transpose2d(x, weight=kernel, stride=2, padding=2)
        out = torch.conv_transpose2d(x, weight=kernel, stride=1)
        return out



class Mem(nn.Module):
    def __init__(self, size=1, keysize=None, outputsize=None):
        super(Mem, self).__init__()
        self.keys = nn.Parameter(torch.randn((size, keysize)))
        self.values = nn.Parameter(torch.randn((size, outputsize)))

    def forward(self, x):
        key = F.interpolate(x, (32, 32))
        # key = torch.sigmoid(key)
        key = key.flatten()
        attention = torch.matmul(self.keys, key)
        attention = torch.softmax(attention, 0)
        attention = torch.reshape(attention, (-1, 1))

        value = self.values * attention
        value = torch.sum(value, 0)
        out = torch.reshape(value, (1, 3, 128, 128))
        return out

class MemA(nn.Module):
    def __init__(self, size=1, keysize=None, outputsize=None):
        super(MemA, self).__init__()
        self.keys = nn.Parameter(torch.randn((size, keysize)))
        self.values = nn.Parameter(torch.randn((size, outputsize)))

    def forward(self, x):
        # key = F.interpolate(x, (32, 32))
        # key = torch.sigmoid(key)
        key = x.flatten()
        attention = torch.matmul(self.keys, key)
        attention = torch.softmax(attention, 0)
        attention = torch.reshape(attention, (-1, 1))

        value = self.values * attention
        value = torch.sum(value, 0)
        # out = torch.reshape(value, (1, 3, 128, 128))
        return value

class MemB(nn.Module):
    def __init__(self, size=1, keysize=None, outputsize=None):
        super(MemB, self).__init__()
        self.keys = nn.Parameter(torch.randn((size, keysize)))
        self.values = nn.Parameter(torch.randn((size, outputsize)))

    def forward(self, x):
        # key = F.interpolate(x, (32, 32))
        # key = torch.sigmoid(key)
        key = x.flatten()
        attention = torch.matmul(self.keys, key)
        attention = torch.softmax(attention, 0)
        attention = torch.reshape(attention, (-1, 1))

        value = self.values * attention
        value = torch.sum(value, 0)
        value = torch.softmax(value, 0)
        # st.write(value.shape)
        # st.stop()
        out = torch.reshape(value, (1, 3, 128, 128))
        return out


class MemC(nn.Module):
    def __init__(self, size=1, keysize=None, outputsize=None):
        super(MemC, self).__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=3, groups=1, kernel_size=(5, 5), stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, groups=1, kernel_size=(5, 5), stride=1, padding=2)
        self.conv3 = nn.Conv2d(in_channels=3, out_channels=3, groups=1, kernel_size=(5, 5), stride=1, padding=2)
        self.keys = nn.Parameter(torch.randn((size, keysize)))
        self.values = nn.Parameter(torch.randn((size, outputsize)))

    def forward(self, x):
        x = self.conv3(self.conv2(self.conv(x)))
        key = F.interpolate(x, (32, 32))
        # key = torch.sigmoid(key)
        key = key.flatten()
        attention = torch.matmul(self.keys, key)
        attention = torch.softmax(attention, 0)
        attention = torch.reshape(attention, (-1, 1))

        value = self.values * attention
        value = torch.sum(value, 0)
        out = torch.reshape(value, (1, 3, 128, 128))
        return out

class Looper(nn.Module):
    def __init__(self, size=1, keysize=None):
        super(Looper, self).__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=3, groups=1, kernel_size=(5, 5), stride=2)
        self.keys = nn.Parameter(torch.randn((size, keysize)))
        self.values = nn.Parameter(torch.randn((size, 3 * 3 * 5 * 5)))

    def forward(self, x, y):
        key = self.conv(x)
        key = torch.sigmoid(key)
        key = F.interpolate(key, (32, 32))
        key = key.flatten()
        # key = torch.tanh(key)
        attention = torch.matmul(self.keys, key)
        attention = torch.softmax(attention, 0)
        attention = torch.reshape(attention, (-1, 1))

        kernel = self.values * attention
        kernel = torch.sum(kernel, 0)
        kernel = torch.reshape(kernel, (3, 3, 5, 5))
        out = torch.conv_transpose2d(x, weight=kernel, stride=2, padding=2)
        # out = torch.sigmoid(out)
        # out = F.interpolate(out, size=(64, 64))
        it = torch.tensor([0])
        # while 1/kornia.psnr_loss(out, y, max_val=1.0) > 0.0:
        while F.mse_loss(out, y) > 0.0:
            # st.write(f'ITERATION: {it} ol: {ol}')
            it += 1
            key = self.conv(out)
            key = torch.sigmoid(key)
            key = F.interpolate(key, (32, 32))
            key = key.flatten()
            # key = torch.tanh(key)
            attention = torch.matmul(self.keys, key)
            attention = torch.softmax(attention, 0)
            attention = torch.reshape(attention, (-1, 1))

            kernel = self.values * attention
            kernel = torch.sum(kernel, 0)
            kernel = torch.reshape(kernel, (3, 3, 5, 5))
            out = torch.conv_transpose2d(out, weight=kernel, stride=2, padding=2)
            if it >= 5:
                break
        # st.write(f'ITERATIONS: {it}')
        return out

upped_image = st.empty()
long_features = st.empty()

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # self.lk1 = LongConv(10, 12288)
        # self.lk2 = LongConv(10, 108300)
        # self.lk3 = LongConv(100, 3072)
        # self.lk4 = LongConv(50, 3072)
        # self.lk5 = LongConv(50, 3072)
        # self.lk6 = LongConv(50, 3072)
        #
        # self.lc = Looper(100, 3072)
        # # self.lm = LongMem(1, 3072)
        # self.lm = LongMem(10, 12288, 3, 3)
        # self.lm2 = LongMem(10, 13872, 3, 3)
        # self.lm3 = LongMem(10, 15552, 3, 3)
        #
        # self.conv = nn.Conv2d(3, 3, (3, 3), stride=1)
        # self.conv2 = nn.Conv2d(3, 3, (3, 3), stride=1)
        # self.conv3 = nn.Conv2d(3, 3, (3, 3), stride=1)
        #
        # self.tconv = nn.ConvTranspose2d(3, 3, (10, 10))
        # self.tconv2 = nn.ConvTranspose2d(3, 3, (10, 10))

        # self.m1 = MemA(10, 75*784, 75 * 15376)
        # self.m2 = MemA(10, 75 * 784, 75 * 15376)
        # self.m3 = MemA(10, 75 * 784, 75 * 15376)
        # self.m4 = MemA(10, 32 * 32 * 3, 128 * 128 * 3)
        # self.m5 = MemA(10, 16 * 16 * 3, 128 * 128 * 3)

        self.mb1 = MemB(10, 128*128*3, 128*128*3)
        # self.m2 = MemC(10, 32*32*3, 128*128*3)
        self.unfold = nn.Unfold(kernel_size=(5, 5))
        self.fold = nn.Fold(kernel_size=(5, 5), output_size=(128, 128), stride=5)
        self.patches = nn.Parameter(torch.randn((1, 75, 15376)))
    def forward(self, input):
        folded = self.mb1(input.flatten())
        folded = torch.reshape(folded, (1, 3, 128, 128))
        folded = folded * input
        # st.write(input.shape)
        # x = F.interpolate(input, size=(32, 32))
        # unfolded = self.unfold(x)
        # st.write(unfolded.shape)
        # unfolded = unfolded.flatten()
        # patches = self.m1(unfolded)
        # patches = torch.reshape(patches, (1, 75, 15376))
        # patches = patches + unfolded
        # folded = self.fold(patches)
        # folded = torch.sigmoid(folded)

        # patches = self.unfold(input)
        # st.write(patches.shape)
        # folded = self.fold(self.patches)
        # # input_ones = torch.ones(input.shape, dtype=input.dtype)
        # # divisor = self.fold(self.unfold(input_ones))
        # # st.write(self.fold(self.unfold(input)) == divisor * input)
        # st.stop()
        # o1 = F.interpolate(input, size=(64, 64))
        # o1 = (2 * o1) - 1
        # o1 = self.m1(o1)
        # o2 = F.interpolate(input, size=(32, 32))
        # o2 = (2 * o2) - 1
        # o2 = self.m2(o2)
        # o3 = F.interpolate(input, size=(16, 16))
        # o3 = (2 * o3) - 1
        # o3 = self.m3(o3)
        # # out = o1 + o2 + o3
        # o1 = torch.sigmoid(o1)
        # # st.write(out.shape)
        # # st.write(input.shape)
        # out = o1 * input.flatten()
        # out = torch.sigmoid(out + o2 + o3)
        # out = torch.reshape(out, (1, 3, 128, 128))

        image = folded.permute(0, 2, 3, 1)
        # image = torch.sigmoid(image)
        upped_image.image(image.detach().cpu().numpy(), width=250, caption='upscaled')
        # st.stop()
        # out = F.interpolate(out, size=(128, 128))
        # out = torch.sigmoid(folded)
        return folded

def load_img(path:str="image.png"):
    # use or not use 4th channel??(transparency)
    image = Image.open(path).convert('RGB')
    # normalize
    img = np.array(image) / 255
    # img = torch.tensor(img)
    return img

def process(image):
    pass

def get_patches(image, kernel=(3, 3), stride=5 , padding=10):
    # image = torch.tensor(image)
    # x_image = image.permute(0, 3, 1, 2)
    # x_image = torch.unsqueeze(x_image, 0)
    # st.write(x_image.shape)
    # x_image = torch.nn.functional.interpolate(x_image, (128, 128))
    patches = torch.nn.functional.unfold(image, kernel_size=kernel, stride=stride, padding=10)
    return patches

def recover(patches, kernel=(3, 3), stride=5, out_shape=(64, 64)):
    return  torch.nn.functional.fold(patches, kernel_size=kernel, stride=stride, padding=10, output_size=out_shape)

def checkDataset(dataset):
    # cols = st.beta_columns(4)
    status_pos = st.empty()
    status = []
    i = 0
    all_patches = Counter()
    for sample_orig in dataset:
        sample = sample_orig / 255
        sample_t = torch.tensor(sample)
        sample_t = torch.reshape(sample_t, (1, 1, 64, 64))
        unfolded = get_patches(sample_t)

        patches = torch.squeeze(unfolded, 0)
        patches.permute(1, 0)
        # st.write(patches.shape)

        folded = recover(unfolded)
        # st.write(patches[0])
        for patch in patches:
            all_patches.update([str(patch)])
        print('*********')
        print(all_patches.most_common(1)[0][1])
        input_ones = torch.ones(sample_t.shape, dtype=sample_t.dtype)
        divisor = recover(get_patches(input_ones))
        folded /= divisor

        flat1 = torch.flatten(sample_t) * 255
        flat2 = torch.flatten(folded) * 255

        array1 = [chr(int(c)) for c in flat1]
        array2 = [chr(int(c)) for c in flat2]

        ok = 0
        bad = 0
        for c1, c2 in zip(array1, array2):
            if c1 == c2:
                ok += 1
            else:
                bad += 1
        status.append([i, ok, bad])
        i += 1
        if i == 10:
            breakpoint()
    status_pos.write(status)
        # st.write(f'BAD: {bad}\tOK: {ok}')
    st.write('END OF CHECK!')

def run_app():
    dataset = WikiDataset(dtype=int)
    st.write('Dataset Loaded!')
    checkDataset(dataset)
    st.stop()
    # comp_col = st.beta_columns(3)
    # image_cols = st.beta_columns(5)
    # out_cols = st.beta_columns(2)
    # text_cols = st.beta_columns(2)
    # p = [st.empty() for x in range(20)]
    # dataset = WikiDataset(dtype=int)
    # p[0].write(f'Dataset length: {len(dataset)}')
    # sample_orig = dataset[10]
    # comp_col[0].text(sample_orig)
    # sample = sample_orig / 255
    # p[1].write(f'Sample length: {len(sample)}')
    # p[2].write(f'Sample Shape: {sample.shape}')
    #
    # sample_t = torch.tensor(sample)
    # sample_t = torch.reshape(sample_t, (1, 1, 64, 64))
    # p[3].write(f'Tensor length: {len(sample_t)}')
    # p[4].write(f'Tensor Shape: {sample_t.shape}')
    # img = sample_t.permute(0, 2, 3, 1).numpy()
    # # p[5].image(img, width=250)
    # image_cols[0].image(img, use_column_width=True)
    # # sample_t *= 127
    # unfolded = get_patches(sample_t)
    # p[6].write(f'patches: {len(unfolded)}')
    # p[7].write(f'patches shape: {unfolded.shape}')
    # folded = recover(unfolded)
    # input_ones = torch.ones(sample_t.shape, dtype=sample_t.dtype)
    # divisor = recover(get_patches(input_ones))
    # folded /= divisor
    # # recov /= torch.max(recov)
    # # while True:
    # p[8].write(f'recov: {len(folded)}')
    # p[9].write(f'recov shape: {folded.shape}')
    # img2 = folded.permute(0, 2, 3, 1).numpy()
    # # p[10].image(img2, width=250)
    # image_cols[1].image(img2, use_column_width=True)
    # flat = torch.flatten(sample_t) * 255
    # out_cols[0].text(str(flat))
    # # TODO: remove the int operation
    # flat = flat.int()
    # # flat = torch.round(flat)
    # array = [chr(int(c)) for c in flat]
    # text1 = ''.join(array)
    # # p[11].text(text1)
    # text_cols[0].text(text1)
    # # flat = torch.flatten(recov-0.0005) * 126
    # # **************************************************************************
    # flat = torch.flatten(folded) * 255
    # out_cols[1].text(str(flat))
    # # TODO: remove the int operation
    # flat = flat.int()
    # # flat = torch.round(flat)
    # array = [chr(int(c)) for c in flat]
    # text2 = ''.join(array)
    # # p[12].text(text2)
    # text_cols[1].text(text2)
    # diff_img = np.abs(img2 - img)
    # image_cols[2].image(diff_img, use_column_width=True)
    # diff = np.sum(diff_img)
    # image_cols[3].write(f'DIFF: {diff}')
    # bad = 0
    # ok = 0
    # for c1,c2 in zip(text1, text2):
    #     if c1 == c2:
    #         ok += 1
    #     else:
    #         bad += 1
    # st.write(f'BAD: {bad}\tOK: {ok}')





    # p_gen = make_patch_generator(sample_t.permute(0, 2, 3, 1).squeeze(0).numpy(), shape=(16, 16))
    # for p, l in p_gen:
    #     # st.write(p)
    #     # st.write(p.shape)
    #     image_cols[4].image(p, width=250)
    #     sleep(2)
    # sleep(2)
    # patches = torch.nn.functional.unfold(sample_t, kernel_size=(1, 32), stride=2, padding=10)
    # recov = torch.nn.functional.fold(patches, kernel_size=(1, 32), stride=2, padding=10, output_size=(1, 1000))
    # recov /= torch.max(recov)
    # st.write(f'Recov Shape: {recov.shape}')
    # st.write(sample_t)
    # st.write(recov)
    # # patches =
    # p[5].write(f'Patches shape: {patches.shape}')
    # patches = torch.squeeze(patches, 0)
    # patches = patches.permute(1, 0)
    # p[6].write(f'Patches shape: {patches.shape}')
    # diff = torch.sum(torch.abs(recov - sample_t))
    # st.write(f'DIFF: {diff}')
    # sample_t = torch.reshape(sample_t, (1000,))
    # recov = torch.reshape(recov, (1000,))
    # sample_t *= 128
    # recov *= 128
    # s = [chr(int(i)) for i in sample_t]
    # r = [chr(int(i)) for i in recov]
    # st.write(''.join(s))
    # st.write(''.join(r))
    # for patch in patches:
    #     st.write(f'Patches shape: {patch.shape}')
    #     st.write(patch)
if __name__ == '__main__':
    run_app()
