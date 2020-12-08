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

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # n = 10
        self.keys = nn.Parameter(torch.randn(500, 4))
        self.keys2 = nn.Parameter(torch.randn(500, 250))
        self.keys3 = nn.Parameter(torch.randn(500, 250))

        self.values = nn.Parameter(torch.randn(500, 250))
        self.values2 = nn.Parameter(torch.randn(500, 250))
        self.values3 = nn.Parameter(torch.randn(500, 2500))


    def forward(self, locations):
        key = torch.flatten(locations)

        attention = torch.matmul(self.keys, key)
        attention = torch.softmax(attention, 0)

        out = torch.matmul(attention, self.values)
        out = torch.relu(out)

        attention = torch.matmul(self.keys2, out)
        attention = torch.softmax(attention, 0)

        out = torch.matmul(attention, self.values2)
        out = torch.relu(out)

        attention = torch.matmul(self.keys3, out)
        attention = torch.softmax(attention, 0)

        out = torch.matmul(attention, self.values3)
        out = torch.sigmoid(out)

        out = torch.reshape(out, (25, 25, 4))
        return out

def load_img(path:str="game.png"):
    image = Image.open(path)
    # normalize
    img = np.array(image) / 255
    img = torch.tensor(img)
    return img

def fun(x):
    return (x * 2) - 1

def make_patch_generator(image, shape, stride=1):
    x_max, y_max, c_max = image.shape
    for x in range(0, x_max - shape[0], stride):
        for y in range(0, y_max - shape[1], stride):
            yield image[x:x + shape[0], y:y + shape[1]], \
                  np.array([fun(x/x_max), fun(y/y_max), fun((x + shape[0])/x_max), fun((y + shape[1])/y_max)])


# @st.cache(suppress_st_warning=True)
def run_app():
    PATH = "state_dict_model.pt"

    cuda = torch.device('cuda')
    cpu = torch.device('cpu')
    net = Net()
    # net.load_state_dict(torch.load(PATH))
    net.to(cuda)
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(net.parameters(), lr=0.00005)

    st.title('image compressor')
    img = load_img('game.png')

    st.write('Original image')
    orig_img_loc = st.image(img.numpy(), width=250, caption='original image: 64 x 64')
    col1, col2 = st.beta_columns(2)
    info = st.empty()
    KERNEL_SIZE = (25, 25)
    STRIDE = 50
    # patch_generator = make_patch_generator(img.numpy(), KERNEL_SIZE, STRIDE)
    slid_win_loc = col1.empty()
    out_loc = col2.empty()
    patch_generator = make_patch_generator(img.numpy(), KERNEL_SIZE, STRIDE)
    peace = [next(patch_generator) for x in range(500)]
    # output_data = st.empty()
    glob_loss_chart = st.empty()
    progress_bar = st.empty()
    # slider_val = st.slider("SPEED", min_value=0.0, max_value=1.0)
    slider = st.empty()
    slider.slider("SPEED", min_value=0.0, max_value=1.0)
    # st.write(str(slider.))
    button = st.button('Click')
    slowdown = False
    losses = deque(maxlen=100)
    first_save = False
    best_loss = None
    st.write(f'dataset size: {len(peace)}')
    while True:
        # patch_generator = make_patch_generator(img.numpy(), KERNEL_SIZE, STRIDE)
        glob_loss = torch.tensor([0.0], device=cuda)
        # progress_bar.progress(0)
        optimizer.zero_grad()
        for patch, location in peace:
            optimizer.zero_grad()
            # optimizer.zero_grad()
            # i += 1
            # progress_bar.progress(i * 10)
            # if (1 - slider_val) != 0:
            #     sleep(1 - slider_val)
            slid_win_loc.image(patch, width=250, caption = f'sliding window patch')
            info.write(f"sliding window patch at \nlocation:  \n{location}  \nshape: {patch.shape}  \nBestLOSS: {best_loss}")
            out = net(torch.tensor(location, device=cuda).float())
            # output_data.table(out.cpu().detach().numpy())
            loss = criterion(out, torch.tensor(patch, device=cuda).float())
            if best_loss is None:
                best_loss = loss * 1
            else:
                if loss < best_loss:
                    best_loss = loss * 1
            loss.backward()
            optimizer.step()
            glob_loss += loss
            if loss.cpu().detach() < 0.0000338:
                slowdown = True
                if first_save is False:
                    torch.save(net.state_dict(), PATH)
                    first_save = True
                    st.write("WEIGHTS SAVED!")
            if slowdown:
                sleep(2)
            out_loc.image(out.cpu().detach().numpy(), width=250, caption=f'LOSS: {loss.detach()}')
        losses.append(glob_loss.cpu().detach().numpy())
        # st.write(str(np.median(losses)))
        glob_loss_data = pd.DataFrame(losses, columns=['global loss'])
        glob_loss_chart.line_chart(glob_loss_data)
        # glob_loss.backward()
        # optimizer.step()


if __name__ == '__main__':

    run_app()
