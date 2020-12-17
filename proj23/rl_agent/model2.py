import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from time import sleep
from collections import deque, Counter

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # n = 10
        self.keys = nn.Parameter(torch.randn(500, 24, dtype=torch.double))
        # self.keys2 = nn.Parameter(torch.randn(500, 250))
        # self.keys3 = nn.Parameter(torch.randn(500, 250))

        self.values = nn.Parameter(torch.randn(500, 4, dtype=torch.double))

        # see how many times a key has been chosen/called
        # self.meta = [0 for x in range(500)]
        self.meta = Counter()
        # self.values2 = nn.Parameter(torch.randn(500, 250))
        # self.values3 = nn.Parameter(torch.randn(500, 2500))


    def forward(self, query):
        attention = torch.matmul(self.keys, query)
        # st.write(f"Key shape: {key.shape}")
        # st.write(f"Keys shape: {self.keys.shape}")
        # st.write(f"Attention shape: {attention.shape}")
        attention = torch.softmax(attention, 0)
        amax = torch.argmax(attention)
        # self.meta[amax] += 1
        self.meta.update(f'{amax}')
        # print(self.meta.most_common(10))
        #
        out = torch.matmul(attention, self.values)
        out = torch.tanh(out)
        return out
        # st.write(f'Values shape: {self.values.shape}')
        # st.write(f'Ovals shape: {out.shape}')
        # out = torch.relu(out)
        #
        # attention = torch.matmul(self.keys2, out)
        # attention = torch.softmax(attention, 0)
        #
        # out = torch.matmul(attention, self.values2)
        # out = torch.relu(out)
        #
        # attention = torch.matmul(self.keys3, out)
        # attention = torch.softmax(attention, 0)
        #
        # out = torch.matmul(attention, self.values3)
        # out = torch.sigmoid(out)
        # st.write(f'Output Shape: {out.shape}')
        #
        # out = torch.reshape(out, (25, 25, 4))
        # st.stop()
        # return out

class Net2(nn.Module):

    def __init__(self):
        super(Net2, self).__init__()
        # n = 10
        self.keys = nn.Parameter(torch.randn(500, 28, dtype=torch.double))
        # self.keys2 = nn.Parameter(torch.randn(500, 250))
        # self.keys3 = nn.Parameter(torch.randn(500, 250))

        self.values = nn.Parameter(torch.randn(500, 1, dtype=torch.double))

        # see how many times a key has been chosen/called
        # self.meta = [0 for x in range(500)]
        self.meta = Counter()
        # self.values2 = nn.Parameter(torch.randn(500, 250))
        # self.values3 = nn.Parameter(torch.randn(500, 2500))


    def forward(self, query):
        attention = torch.matmul(self.keys, query)
        # st.write(f"Key shape: {key.shape}")
        # st.write(f"Keys shape: {self.keys.shape}")
        # st.write(f"Attention shape: {attention.shape}")
        attention = torch.softmax(attention, 0)
        amax = torch.argmax(attention)
        # self.meta[amax] += 1
        self.meta.update(f'{amax}')
        # print(self.meta.most_common(10))
        #
        out = torch.matmul(attention, self.values)
        # out = torch.tanh(out)
        return out
        # st.write(f'Values shape: {self.values.shape}')
        # st.write(f'Ovals shape: {out.shape}')
        # out = torch.relu(out)
        #
        # attention = torch.matmul(self.keys2, out)
        # attention = torch.softmax(attention, 0)
        #
        # out = torch.matmul(attention, self.values2)
        # out = torch.relu(out)
        #
        # attention = torch.matmul(self.keys3, out)
        # attention = torch.softmax(attention, 0)
        #
        # out = torch.matmul(attention, self.values3)
        # out = torch.sigmoid(out)
        # st.write(f'Output Shape: {out.shape}')
        #
        # out = torch.reshape(out, (25, 25, 4))
        # st.stop()
        # return out