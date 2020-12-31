import streamlit as st

import torch
from torch import nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.memory = nn.Parameter(torch.rand(30*30))

    def forward(self, x):
        x = self.memory * x
        x = torch.reshape(x, (30, 30))
        x = x.clamp(0, 1)
        return x

net = Net()
image_tensor = torch.rand((30, 30))
image = image_tensor.detach().numpy()
st.image(image, caption='input image', width=200)
x_inp = torch.ones(1)
out = net(x_inp)
out_image = out.detach().numpy()
st.image(out_image, caption='image from memory', width=200)
# out = net(image_tensor)
# st.write(out)