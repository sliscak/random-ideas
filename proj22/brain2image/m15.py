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
    def __init__(self,
                 path='C:\\Users\\Admin\\Downloads\\mindbigdata-imagenet-in-v1.0\\MindBigData-Imagenet-v1.0-Imgs\\',
                 testing=False):
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
        if not testing:
            self.data = self.data[0:3000]
    # def load_images(self, path):

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        metaimage = self.data[idx]
        x = metaimage.data
        y = torch.tensor([self.categories.index(metaimage.category)])
        # is Y a label?
        sample = x, y
        return sample

def make_kernel(in_channels, out_channels, groups, kernel_size):
    return nn.Parameter(torch.Tensor(
        out_channels, in_channels // groups, *kernel_size))

class Net(nn.Module):

    def __init__(self, kernel_loc):
        super(Net, self).__init__()
        # n = 10
        # output is [1, 1, 30, 30]
        self.kernel_loc = kernel_loc
        self.conv = nn.Conv2d(in_channels=4, out_channels=1, groups=1, kernel_size=(5, 5), stride=2)

        self.keys = nn.Parameter(torch.randn((50,900)))
        #kernel shape is 1x4x5x5
        self.values = nn.Parameter(torch.randn((50, 1*4*5*5)))

        # self.kernel = make_kernel(in_channels = 4, out_channels = 1, groups = 1, kernel_size = [5,5])
        # self.keys =
        # self.kernel2 = make_kernel(in_channels= 4, out_channels = 1, groups = 1, kernel_size= [5, 5])
        # self.kernel = nn.Parameter(torch.Tensor(
        #         out_channels, in_channels // groups, *kernel_size))
        # self.values = nn.Parameter(torch.randn(50, 250))

        # self.keys2 = nn.Parameter(torch.randn(500, 250))
        # self.values2 = nn.Parameter(torch.randn(500, 569))
        #
        self.seq = nn.Sequential(
            nn.Linear(900, 450),
            nn.ReLU(),
            nn.Linear(450, 569),
            # nn.Softmax(),
        )
        # self.keys = nn.Parameter(torch.randn(500, 64*64*4))
        # self.keys2 = nn.Parameter(torch.randn(500, 250))
        # # self.keys3 = nn.Parameter(torch.randn(50, 500))
        #
        # self.values = nn.Parameter(torch.randn(500, 250))
        # self.values2 = nn.Parameter(torch.randn(500, 569))
        # # self.values3 = nn.Parameter(torch.randn(50, 2500))


    def forward(self, image):
        image = (2 * image) - 1
        image = image.permute(2, 1, 0)
        image = torch.unsqueeze(image, 0)
        # st.write(f'IMAGE SHAPE: {image.shape}')
        # key = 1*1*30*30 == 900 after flatten
        key = self.conv(image).flatten()
        # st.write(f'KEY SHAPE: {key.shape}')
        # st.write(f'KERNEL SHAPE: {self.kernel.shape}')
        attention = torch.matmul(self.keys, key)
        attention = torch.softmax(attention, 0)
        attention = torch.reshape(attention, (-1, 1))
        # st.write(f'ATTENTION SHAPE: {attention.shape}')
        # st.write(f'VALUES SHAPE: {self.values.shape}')

        kernel = self.values * attention
        kernel = torch.sum(kernel, 0)
        kernel = torch.reshape(kernel, (1, 4, 5, 5))
        pkernel = torch.sigmoid(torch.squeeze(kernel, 0).permute(2,1,0))
        pkernel = pkernel.detach().cpu().numpy()
        self.kernel_loc.image(pkernel, width=200)
        # TODO: add actiovation function here(to kernel)

        kernel = torch.sigmoid(kernel)
        out = torch.conv2d(image, weight=kernel, stride=2)
        out = torch.flatten(out)
        # Out Shape: torch.Size([1, 1, 30, 30])
        # st.write(f'key2 Shape: {key.shape}')
        # TODO: dont use softmax on output when using crossentropy loss function
        out = self.seq(out)

        return out



def run_app():
    # GUI
    epoch_loc = st.empty()
    prog_bar = st.empty()
    loss_loc = st.empty()
    global_loss_loc = st.empty()
    col1, col2, col3 = st.beta_columns(3)
    img_loc = col1.empty()
    kernel_loc = col2.empty()
    stats_loc = col3.empty()
    image_meta_loc = st.empty()
    right_chart = st.empty()
    loss_chart = st.empty()
    glob_loss_chart = st.empty()
    # right_chart = st.empty()
    test_progress_bar = st.empty()
    testing_chart = st.empty()
    test_stats = st.empty()

    PATH = 'model_TYPE2.pt'
    cuda = torch.device('cuda')
    cpu = torch.device('cpu')
    net = Net(kernel_loc)
    try:
        net.load_state_dict(torch.load(PATH))
    except Exception:
        pass
    net.to(cuda)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(net.parameters(), lr=0.00001)

    dataset = EegDataset()

    losses = deque(maxlen=100)
    global_losses = deque(maxlen=100)
    right_list = deque(maxlen=100)
    wrong_list = deque(maxlen=100)
    EPOCHS = 10
    for epoch in range(EPOCHS):
        i = 1
        epoch_loc.write(f"EPOCH:\t{epoch}/{EPOCHS-1}")
        global_loss = torch.tensor([0.0], device=cuda)
        optimizer.zero_grad()
        right = 0
        wrong = 0
        def collate_wrapper(samples):
            return samples
        #TODO: confirm that shuffle works
        train_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_wrapper)
        for batch in train_loader:
            optimizer.zero_grad()
            loss = torch.tensor([0.0], device=cuda)
            for sample in batch:
                x, y = sample
                prog_bar.progress(i / len(dataset))
                i += 1
                img_loc.image(x.numpy(), width=200)
                # image_meta_loc.write(f"ID:\t{image.id}  \nCategory:\t{image.category}")
                out = net(x.cuda().float()).unsqueeze(0)
                out_id = torch.argmax(out.detach().cpu(), 1)
                target_id = y.cuda()
                # target = torch.tensor([dataset.categories.index(image.category)]).cuda()
                # stats_loc.write(f"OUTPUT:\t{torch.argmax(out.detach().cpu(), 1)}  \nTARGET:\t{target.detach().cpu()}")
                stats_loc.write(f"OUTPUT:\t{out_id}  \nTARGET:\t{target_id}")

                loss += criterion(out, target_id)
                if out_id == target_id.detach().cpu():
                    right += 1
                    # use len(dataset.categories) ; i want to divide by the number of categories.
                    loss = loss * (1 / len(dataset))
                else:
                    wrong += 1
                    loss = loss * 1

            losses.append(loss.detach().cpu().numpy())
            loss_chart.line_chart(
                pd.DataFrame(losses, columns=['loss',])
            )
            global_loss += loss
            loss_loc.write(f"LOSS:\t{loss.detach().cpu()}  \nRIGHT:\t{right}/{len(dataset)}  \nWRONG:\t{wrong}/{len(dataset)}")
            loss.backward()
            optimizer.step()
        right_list.append(right)
        wrong_list.append(wrong)
        rc_data = pd.DataFrame(np.array([[r,w] for r,w in zip(right_list, wrong_list)]), columns=['right', 'wrong'])
        right_chart.line_chart(rc_data)
        # wc_data = pd.DataFrame(np.array(wrong_list), columns=['wrong',])
        global_loss_loc.write(f"GLOBAL LOSS:\t{global_loss.detach().cpu()}  \nGLOB AVERAGE LOSS:\t{global_loss.detach().cpu()/len(dataset)}")
        global_losses.append(global_loss.detach().cpu().numpy())
        glob_loss_chart.line_chart(
            pd.DataFrame(global_losses, columns=['global_loss', ])
        )
        # global_loss.backward()
        # optimizer.step()
    # # TESTING PHASE:
    dataset = EegDataset(testing=True)
    def collate_wrapper(samples):
        return samples

    test_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_wrapper)
    right = 0
    wrong = 0
    st.write('TESTING!!!!!!!!!!!!!!!!!!!/EVALUATING????')
    i = 1
    with torch.no_grad():
        for batch in test_loader:
            for sample in batch:
                x,y = sample
                test_progress_bar.progress(i / len(dataset))
                i += 1
                out = net(x.cuda().float())
                out_id = torch.argmax(out.detach().cpu(), 0)
                target_id = y
                if out_id == target_id:
                    right += 1
                else:
                    wrong += 1
                test_stats.write(f'RIGHT: {right}/{len(dataset)}  \nWRONG: {wrong}/{len(dataset)}')
    torch.save(net.state_dict(), PATH)
if __name__ == '__main__':
    run_app()
    st.write('##Model saved!')