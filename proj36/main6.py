import torch
import streamlit as st
from torch import nn
from torch.optim import AdamW


class Mem(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.param = nn.Parameter(torch.randn(output_size, input_size), requires_grad=True)

    def forward(self, x):
        x = x.unsqueeze(0)
        o = torch.cosine_similarity(self.param, x)
        return o

class Net(torch.nn.Module):
    """"
        Multi custom layer neural network
    """
    def __init__(self, input_size, output_size):
        super().__init__()
        self.layer1 = Mem(input_size, 10)
        self.layer2 = Mem(10, 10)
        self.layer3 = Mem(10, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

placeholders_ = [[col.empty() for col in st.beta_columns(1)] for x in range(5)]
placeholders = [[col.empty() for col in st.beta_columns(1)] for x in range(5)]
net = Net(12, 12)
optimizer = AdamW(net.parameters(), lr=0.001)
criterion = nn.MSELoss()

x_ = [torch.tensor([
    [0, 1, 0, 0],
    [1, 1, 1, 1],
    [0, 1, 0, 0],
], dtype=float), torch.tensor([
    [0, 0, 1, 0],
    [1, 1, 1, 1],
    [0, 0, 1, 0],
], dtype=float)]
y_ = [torch.tensor([
    [1, 0, 0, 0],
    [1, 0, 0, 0],
    [1, 0, 0, 0],
], dtype=float), torch.tensor([
    [0, 0, 0, 1],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
], dtype=float)]

while True:
    for x,y in zip(x_, y_):
        placeholders_[0][0].write(x.detach().numpy())
        placeholders_[1][0].write(y.detach().numpy())
        output = net(x.flatten()).reshape((3, 4))
        loss = criterion(output, y)
        placeholders_[2][0].write(output.detach().numpy())
        print(f'Loss: {loss.detach()}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        params = net.parameters()
        # print(list(enumerate(params)))
        for i, param in enumerate(params):
            placeholders[i][0].write(param.detach().numpy())
        # print(params)
        # exit()
        # placeholders[0][0].write()
