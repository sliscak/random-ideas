import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import os
import kornia
from PIL import Image
from time import sleep
from collections import deque

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

def get_patches(image, kernel=(25, 25), stride=(25, 25)):
    image = torch.tensor(image)
    x_image = image.permute(2, 0, 1)
    x_image = torch.unsqueeze(x_image, 0)
    st.write(x_image.shape)
    x_image = torch.nn.functional.interpolate(x_image, (128, 128))
    patches = torch.nn.functional.unfold(x_image, kernel_size=kernel, stride=stride)
    return patches

def run_app():
    p1 = st.empty()
    p2 = st.empty()
    p3 = st.empty()
    p4 = st.empty()
    p5 = st.empty()
    # t = torch.zeros(5, 5)
    img = load_img("img.png")
    st.write(img.shape)
    # mask = np.zeros(img.shape[0:2])
    mask = np.zeros((img.shape[0]*2, img.shape[1]*2, img.shape[2]), dtype=float)
    st.image(img)
    for y in range(mask.shape[1]):
        if y%2 == 0:
            for x in range(mask.shape[0]):
                if x % 2 == 0:
                    # print(x,y)
                    # mask[x][y] = [1, 1, 1, 1]
                    mask[x][y] = img[int(x/2)][int(y/2)]
                    # print(img[int(x/2)][int(y/2)])

    mask2 = np.zeros((img.shape[0] * 2, img.shape[1] * 2, img.shape[2]), dtype=float)
    # st.image(img)
    for y in range(mask2.shape[1]):
        if (y+1) % 2 == 0:
            for x in range(mask2.shape[0]):
                if (x+1) % 2 == 0:
                    # print(x,y)
                    # mask[x][y] = [1, 1, 1, 1]
                    mask2[x][y] = img[int((x-1) / 2)][int((y-1) / 2)]
    p1.image(mask)
    p2.image(mask2)
    tensor = torch.unsqueeze(torch.tensor(mask2), 0)
    tensor = tensor.permute(0, 3, 1, 2)
    st.write(tensor.shape)
    new_image = torch.nn.functional.interpolate(tensor, (img.shape[0], img.shape[1]))
    new_image = torch.squeeze(new_image, 0).permute(1, 2, 0).numpy()
    p3.image(new_image, 'new image')
    diff = np.sum(np.abs(new_image - img))
    p4.write(diff)
    img2 = np.random.random(img.shape)

    p5.image(img2, 250)
    st.write('HERE!!!!!!!!!!!!')
    best_diff = None
    best_image = None
    # while True:
    #     for y in range(mask2.shape[1]):
    #         if y % 2 == 0:
    #             for x in range(mask2.shape[0]):
    #                 if x % 2 == 0:
    #                     # st.write(mask2)
    #                     mask2[x][y] = np.random.random((4,))
    #                     # st.write(np.random.randint(0, 255, (3,)))
    #     p1.image(mask)
    #     p2.image(mask2)
    #     tensor = torch.unsqueeze(torch.tensor(mask2), 0)
    #     tensor = tensor.permute(0, 3, 1, 2)
    #     st.write(tensor.shape)
    #     new_image = torch.nn.functional.interpolate(tensor, (img.shape[0], img.shape[1]))
    #     new_image = torch.squeeze(new_image, 0).permute(1, 2, 0).numpy()
    #     p3.image(new_image)
    #     diff = np.sum(np.abs(new_image - img))
    #     p4.write(diff)
    #     if best_diff is None:
    #         best_diff = diff * 1
    #         best_image = new_image * 1
    #     if diff > best_diff:
    #         new_image = best_image * 1
    #     else:
    #         best_diff = diff * 1
    #         best_image = new_image * 1
    #     sleep(1)

    # out = img * mask
    # st.image(out)
    # diff = np.abs(out - img)
    # st.image(diff)
    # st.image(diff + out)
    # st.write(mask)
    # print(t)
    # # dataset_textbox = st.sidebar.text_input('dataset path', value='C:\\Users\\Admin\\Downloads\\i\\n01514859\\')
    #
    # DATASET_PATH = st.text_input('DATASET PATH', value='C:\\Users\\Admin\\Downloads\\i\\n01514859\\')
    # patch_slider_loc = st.empty()
    # patch_loc = st.empty()
    # epoch_loc = st.empty()
    # prog_bar = st.empty()
    # loss_loc = st.empty()
    # global_loss_loc = st.empty()
    # loss_chart = st.empty()
    # glob_loss_chart = st.empty()
    # row0 = st.empty()
    # row1 = st.empty()
    # row2 = st.empty()
    # row3 = st.empty()
    # row4 = st.empty()
    # row5 = st.empty()
    #
    # # st.stop()
    # PATH = "upscaler.pt"
    # net = Net()
    # # too lazy to detect if the file exits.
    # try:
    #     net.load_state_dict(torch.load(PATH))
    #     st.write('MODEL LOADED!')
    # except Exception:
    #     pass
    # cuda = torch.device('cuda')
    # net.to(cuda)
    # # criterion = nn.CrossEntropyLoss()
    # # criterion = nn.MSELoss()
    # criterion = kornia.losses.PSNRLoss(1.0)
    # LEARNING_RATE = 0.01
    # optimizer = optim.AdamW(net.parameters(), lr=LEARNING_RATE)
    # # st.title('image upscaler')
    # img = load_img('image.png')
    #
    # losses = deque(maxlen=100)
    # global_losses = deque(maxlen=100)
    # EPOCHS = 500
    # BATCH_SIZE = 1
    #
    # dataset = ImageDataset(path=DATASET_PATH)
    # sample = dataset[0]
    # x_image, y_image = sample
    # x_image = torch.tensor(x_image)
    # x_image = x_image.permute(2, 0, 1)
    # x_image = torch.unsqueeze(x_image, 0)
    # st.write(x_image.shape)
    # x_image = torch.nn.functional.interpolate(x_image, (128, 128))
    # patches = torch.nn.functional.unfold(x_image, (25, 25), stride=(25, 25))
    # st.write(patches.shape)
    # # st.stop()
    # # st.stop()
    # # patches = patches.permute(0, 2, 1)
    # # st.write(patches.shape)
    # patch_slider = st.slider('patch', min_value=0, max_value=patches.shape[2]-1)
    # patches = patches.permute(0, 2, 1)
    # patch = patches[(0, patch_slider)].reshape(25, 25, 3)
    # st.image(caption=f'PATCH:  \nMIN: {torch.min(patch)}  \n MAX: {torch.max(patch)}', image=patch.numpy(), width=100)
    # # patches = torch.nn.functional.unfold(x_image, (5, 5))
    # # patch_slider_loc = st.slider(min_value=0, max_value=800)
    # x_image = torch.squeeze(x_image, 0)
    # x_image = x_image.permute(1, 2, 0)
    # st.image(x_image.numpy(), width=250)
    #
    #

if __name__ == '__main__':
    run_app()
