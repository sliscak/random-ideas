import pandas
import torch
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
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
        self.layer1 = Mem(input_size, 1500)
        self.layer2 = Mem(1500, 1)
        # self.layer3 = Mem(1, output_size)
        # self.layer1 = nn.Linear(input_size, 5)
        # self.layer2 = nn.Linear(5, 1)
        self.layer3 = nn.Linear(1, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

# placeholders_ = [[col.empty() for col in st.beta_columns(1)] for x in range(5)]
placeholders = [[col.empty() for col in st.beta_columns(1)] for x in range(5)]
net = Net(12, 12)
optimizer = AdamW(net.parameters(), lr=0.001)
criterion = nn.MSELoss()

x_ = [torch.tensor([
    [0, 1, 0, 0],
    [1, 1, 1, 1],
    [0, 1, 0, 0],
], dtype=torch.float), torch.tensor([
    [0, 0, 1, 0],
    [1, 1, 1, 1],
    [0, 0, 1, 0],
], dtype=torch.float)]
y_ = [torch.tensor([
    [1, 0.5, 0.5, 1],
    [1, 0, 0, 0],
    [1, 0, 0, 0],
], dtype=torch.float), torch.tensor([
    [0, 0, 0, 1],
    [0, 0, 0, 1],
    [1, 0.5, 0.5, 1],
], dtype=torch.float)]

iteration = 0
while True:
    loss = torch.zeros(1, dtype=torch.float)
    for x,y in zip(x_, y_):
        x_df = pd.DataFrame(data=x.detach().numpy())
        # x_df = pd.DataFrame(x_df.groupby(by=[0,2]))
        # x_df = x_df[0:2]
        # x_df = pd.DataFrame(data=x_df)
        x_df = x_df.style.background_gradient(cmap='Greys', axis=None)
        # x_df = x_df.style.background_gradient(cmap='Greys', subset=slice(0,3,1))
        # x_df = x_df.style.background_gradient(cmap='Greys', axis=None, subset=slice(0,10))

        placeholders[1][0].write(x_df)

        y_df = pd.DataFrame(data=y.detach().numpy())
        y_df = y_df.style.background_gradient(cmap='Greys', axis=None)
        placeholders[2][0].write(y_df)
        output = net(x.flatten()).reshape((3, 4))
        loss += criterion(output, y)


        out_df = pd.DataFrame(data=output.detach().numpy())
        out_df = out_df.style.background_gradient(cmap='Greys', axis=None)
        placeholders[3][0].write(out_df)


        # params = net.parameters()
        # # print(list(enumerate(params)))
        # for i, param in enumerate(params):
        #     if i == 0:
        #         p_df = pd.DataFrame(data=param.reshape(-1, 3).detach().numpy())
        #         p_df = p_df.style.background_gradient(cmap='Greys', axis=None)
        #         placeholders[i][0].write(p_df)
        #     else:
        #         placeholders[i][0].write(param.detach().numpy())

    optimizer.zero_grad()
    # meta_df = pd.DataFrame(data={'loss':loss.detach().numpy(), 'iteration': iteration}, index=pd.Index(['loss', 'iteration']))
    meta_df = pd.DataFrame(data=np.array([loss.detach().numpy(), iteration]), index=pd.Index(['loss', 'iteration']))
    placeholders[0][0].write(meta_df)
    # placeholders_[0][1].write(f'LOSS: {loss.detach().numpy()}')
    loss.backward()
    optimizer.step()
    print(f'Loss: {loss.detach()}')
    iteration += 1
