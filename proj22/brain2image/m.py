import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from time import sleep
from collections import deque
from torchvision import transforms, utils #,datasets
from torch.utils.data import Dataset, DataLoader
from collections import namedtuple
import os


#first load data
# dataset = datasets.ImageFolder('C:\\Users\\Admin\\Downloads\\mindbigdata-imagenet-in-v1.0\\MindBigData-Imagenet-v1.0-Imgs')
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
# images, labels = next(iter(dataloader))
# # helper.imshow(images[0], normalize=False)
# st.image(images[0])

def load_img(path):
    image = Image.open(path)
    # normalize
    img = np.array(image) / 255
    img = torch.tensor(img)
    return img


# 569 categories
# image shape: 64x64x4
class EegDataset(Dataset):
    def __init__(self, path='C:\\Users\\Admin\\Downloads\\mindbigdata-imagenet-in-v1.0\\MindBigData-Imagenet-v1.0-Imgs\\'):
        self.image_folder_path = path
        self.data = []
        self.categories = set()

        # TODO: Load on the fly
        for root, dirs, files in os.walk(path, topdown=False):
            for image_name in files:
                if '.png' in image_name:
                    MetaImage = namedtuple('MetaImage', ['category', 'id', 'data'])
                    category, id = image_name.split('_')[3:5]
                    image_path = self.image_folder_path + image_name
                    image = load_img(image_path)
                    meta_img = MetaImage(category=category, id=id, data=image)
                    self.data.append(meta_img)
                    self.categories.add(category)
        self.categories = list(self.categories)
        self.data = self.data[0:100]
    # def load_images(self, path):

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        # y_ground =
        return sample


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # n = 10
        self.keys = nn.Parameter(torch.randn(500, 64*64*4))
        self.keys2 = nn.Parameter(torch.randn(500, 250))
        # self.keys3 = nn.Parameter(torch.randn(50, 500))

        self.values = nn.Parameter(torch.randn(500, 250))
        self.values2 = nn.Parameter(torch.randn(500, 569))
        # self.values3 = nn.Parameter(torch.randn(50, 2500))


    def forward(self, image):
        image = (2 * image) - 1
        key = torch.flatten(image)

        attention = torch.matmul(self.keys, key)
        attention = torch.softmax(attention, 0)

        out = torch.matmul(attention, self.values)
        out = torch.relu(out)

        attention = torch.matmul(self.keys2, out)
        attention = torch.softmax(attention, 0)

        out = torch.matmul(attention, self.values2)
        out = torch.softmax(out, 0)
        # out = 1-torch.softmax(out, 0)

        # out = torch.reshape(out, (569,))
        return out



def run_app():
    # GUI
    epoch_loc = st.empty()
    prog_bar = st.empty()
    loss_loc = st.empty()
    global_loss_loc = st.empty()
    col1, col2 = st.beta_columns(2)
    img_loc = col1.empty()
    stats_loc = col2.empty()
    image_meta_loc = st.empty()
    loss_chart = st.empty()
    glob_loss_chart = st.empty()

    cuda = torch.device('cuda')
    cpu = torch.device('cpu')
    net = Net()
    net.to(cuda)

    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(net.parameters(), lr=0.001)

    dataset = EegDataset()

    EPOCHS = 100
    losses = deque(maxlen=100)
    global_losses = deque(maxlen=100)
    for epoch in range(EPOCHS):
        i = 1
        epoch_loc.write(f"EPOCH:\t{epoch}/{EPOCHS-1}")
        global_loss = torch.tensor([0.0], device=cuda)
        optimizer.zero_grad()
        for image in dataset:
            prog_bar.progress(i / len(dataset))
            i += 1
            optimizer.zero_grad()
            x = image.data
            img_loc.image(image.data.numpy(), width=200)
            image_meta_loc.write(f"ID:\t{image.id}  \nCategory:\t{image.category}")
            out = net(x.cuda().float()).unsqueeze(0)
            target_id = dataset.categories.index(image.category)
            target = torch.zeros(len(dataset.categories))
            target[target_id] = 1
            target = target.cuda().float()
            # target = torch.tensor([dataset.categories.index(image.category)]).cuda()
            # stats_loc.write(f"OUTPUT:\t{torch.argmax(out.detach().cpu(), 1)}  \nTARGET:\t{target.detach().cpu()}")
            stats_loc.write(f"OUTPUT:\t{torch.argmax(out.detach().cpu(), 1)}  \nTARGET:\t{target_id}")
            # print(target.shape)
            loss = criterion(out, target)


            losses.append(loss.detach().cpu().numpy())
            loss_chart.line_chart(
                pd.DataFrame(losses, columns=['loss',])
            )
            global_loss += loss
            loss.backward()
            optimizer.step()
            loss_loc.write(f"LOSS:\t{loss.detach().cpu()}")
            global_loss_loc.write(f"GLOBAL LOSS:\t{global_loss.detach().cpu()}")
            # print(loss)
        global_losses.append(global_loss.detach().cpu().numpy())
        glob_loss_chart.line_chart(
            pd.DataFrame(global_losses, columns=['global_loss', ])
        )
        # global_loss.backward()
        # optimizer.step()

if __name__ == '__main__':
    run_app()