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
        print(self.meta.most_common(10))
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
        self.keys = nn.Parameter(torch.randn(500, 76, dtype=torch.double))
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

class Net3(nn.Module):

    def __init__(self):
        super(Net3, self).__init__()
        # n = 10
        self.keys = nn.Parameter(torch.randn(500, 36879, dtype=torch.double))
        # self.keys2 = nn.Parameter(torch.randn(500, 250))
        # self.keys3 = nn.Parameter(torch.randn(500, 250))

        # 36879    ,  12288
        self.values = nn.Parameter(torch.randn(500, 36879, dtype=torch.double))

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
        out = torch.sigmoid(out)
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

class NeuralDictionaryV4(nn.Module):

    def __init__(self):
        super(NeuralDictionaryV4, self).__init__()
        # 500 keys each of size 100, so the query needs to be of size 100
        self.keys = nn.Parameter(torch.randn(500, 76, dtype=torch.double))

        # 500 values each of size 4, the output of the model will be of size 4
        self.values = nn.Parameter(torch.randn(500, 1, dtype=torch.double))

        # to track and later see how many times a key has been chosen as the most important one(the key with the highest confidence)
        self.meta = Counter()

    def forward(self, query):
        # attention = torch.matmul(self.keys, query)
        query = torch.unsqueeze(query, 0)
        query = query.repeat(500, 1) # now query has shape (500,100)
        attention = torch.abs(self.keys - query) # computes absolute difference per element , (maybe later try euclidean distance or cosine similarity)
        attention = -torch.sum(attention, 1) # computes absolute difference per key, that is one key has one value, and makes it negative because of the following softmax operation
        attention = torch.softmax(attention, 0) # compute the probabilities from the differences
        out = torch.matmul(attention, self.values)
        # use a activation function here if you want, like sigmoid, but that depends on the task, the output range we need
        # out = torch.tanh(out)

        amax = torch.argmax(attention)
        self.meta.update(f'{amax}')
        # print(self.meta.most_common(10))

        return out

# class NeuralDictionaryV5(nn.Module):
#
#     def __init__(self):
#         super(NeuralDictionaryV5, self).__init__()
#         # 500 keys each of size 100, so the query needs to be of size 100
#         self.keys = nn.Parameter(torch.randn(500, 76, dtype=torch.double))
#
#         # 500 values each of size 4, the output of the model will be of size 4
#         self.values = nn.Parameter(torch.randn(500, 1, dtype=torch.double))
#
#         # to track and later see how many times a key has been chosen as the most important one(the key with the highest confidence)
#         self.meta = Counter()
#
#     def forward(self, query):
#         # attention = torch.matmul(self.keys, query)
#         query = torch.unsqueeze(query, 0)
#         query = query.repeat(500, 1) # now query has shape (500,100)
#         attention = torch.abs(self.keys - query) # computes absolute difference per element , (maybe later try euclidean distance or cosine similarity)
#         attention = torch.sum(attention, 1) # computes absolute difference per key, that is one key has one value, and makes it negative because of the following softmax operation
#         # attention = torch.softmax(attention, 0) # compute the probabilities from the differences
#         # out = torch.matmul(attention, self.values)
#         # use a activation function here if you want, like sigmoid, but that depends on the task, the output range we need
#         # out = torch.sigmoid(out)
#         out = attention
#         # amax = torch.argmax(attention)
#         # self.meta.update(f'{amax}')
#         # print(self.meta.most_common(10))
#
#         return out

class NeuralDictionaryV5(nn.Module):
    # Compares all keys with the query, computes the absolute differences per element between key and query, sums up the differences per key,
    #    then uses softmax to compute the probabilities per key and matrix multiplies the probabilities with the values.

    def __init__(self, in_features: int, out_features: int, capacity: int):
        # capacity,C, represents the number of key-value pairs.
        self.capacity = capacity
        super(NeuralDictionaryV5, self).__init__()
        # C keys each of size {in_features}, so the query needs to be of size {in_features}
        # for example 500 keys each of size 100, so the query needs to be of size 100
        self.keys = nn.Parameter(torch.randn(capacity, in_features, dtype=torch.float))

        # C values each of size {out_features}, the output of the model will be of size {out_features}
        # for example 500 values each of size 4, the output of the model will be of size 4
        self.values = nn.Parameter(torch.randn(capacity, out_features, dtype=torch.float))

        # to track and later see how many times a key has been chosen as the most important one(the key with the highest confidence)
        self.meta = Counter()

    def forward(self, query):
        # attention = torch.matmul(self.keys, query)
        query = torch.unsqueeze(query, 0)
        query = query.repeat(self.capacity, 1)  # now query has shape (500,100)
        attention = torch.abs(self.keys - query)  # computes absolute difference per element , (maybe later try euclidean distance or cosine similarity)
        attention = -torch.sum(attention,1)  # computes the sum of absolute differences per key, that is one key has one value, and makes it negative because of the following softmax operation
        attention = torch.softmax(attention, 0)  # compute the probabilities from the differences
        out = torch.matmul(attention, self.values)
        # use a activation function here if you want, like sigmoid, but that depends on the task, the output range we need
        # out = torch.sigmoid(out)

        amax = torch.argmax(attention)
        self.meta.update(f'{amax}')
        # print(self.meta.most_common(10))

        return out

class NeuralDictionaryV5Double(nn.Module):
    # Compares all keys with the query, computes the absolute differences per element between key and query, sums up the differences per key,
    #    then uses softmax to compute the probabilities per key and matrix multiplies the probabilities with the values.

    def __init__(self, in_features: int, out_features: int, capacity: int):
        # capacity,C, represents the number of key-value pairs.
        self.capacity = capacity
        super(NeuralDictionaryV5Double, self).__init__()
        # {C} keys each of size {in_features}, so the query needs to be of size {in_features}
        # for example 500 keys each of size 100, so the query needs to be of size 100
        self.keys = nn.Parameter(torch.randn(capacity, in_features, dtype=torch.double))

        # {C} values each of size {out_features}, the output of the model will be of size {out_features}
        # for example 500 values each of size 4, the output of the model will be of size 4
        self.values = nn.Parameter(torch.randn(capacity, out_features, dtype=torch.double))

        # to track and later see how many times a key has been chosen as the most important one(the key with the highest confidence)
        self.meta = Counter()

    def forward(self, query):
        # attention = torch.matmul(self.keys, query)
        query = torch.unsqueeze(query, 0)
        query = query.repeat(self.capacity, 1)  # now query has shape (500,100)
        attention = torch.abs(self.keys - query)  # computes absolute difference per element , (maybe later try euclidean distance or cosine similarity)
        attention = -torch.sum(attention,1)  # computes the sum of absolute differences per key, that is one key has one value, and makes it negative because of the following softmax operation
        attention = torch.softmax(attention, 0)  # compute the probabilities from the differences
        out = torch.matmul(attention, self.values)
        # use a activation function here if you want, like sigmoid, but that depends on the task, the output range we need
        # out = torch.sigmoid(out)

        amax = torch.argmax(attention)
        self.meta.update(f'{amax}')
        # print(self.meta.most_common(10))

        return out

class NeuralDictionaryV5XDouble(nn.Module):
    # Compares all keys with the query, computes the absolute differences per element between key and query, sums up the differences per key,
    #    then uses softmax to compute the probabilities per key and matrix multiplies the probabilities with the values.

    def __init__(self, in_features: int, out_features: int, capacity: int):
        # capacity,C, represents the number of key-value pairs.
        self.capacity = capacity
        super(NeuralDictionaryV5XDouble, self).__init__()
        # {C} keys each of size {in_features}, so the query needs to be of size {in_features}
        # for example 500 keys each of size 100, so the query needs to be of size 100
        self.keys = nn.Parameter(torch.randn(capacity, in_features, dtype=torch.double))

        # {C} values each of size {out_features}, the output of the model will be of size {out_features}
        # for example 500 values each of size 4, the output of the model will be of size 4
        self.values = nn.Parameter(torch.randn(capacity, out_features, dtype=torch.double))

        # to track and later see how many times a key has been chosen as the most important one(the key with the highest confidence)
        self.meta = Counter()

    def forward(self, query):
        # attention = torch.matmul(self.keys, query)
        attention = (self.keys < query) * self.keys  # computes absolute difference per element , (maybe later try euclidean distance or cosine similarity)
        attention = attention.sum(1)  # computes the sum of absolute differences per key, that is one key has one value, and makes it negative because of the following softmax operation
        attention = torch.softmax(attention, 0)  # compute the probabilities from the differences
        out = torch.matmul(attention, self.values)
        # use a activation function here if you want, like sigmoid, but that depends on the task, the output range we need
        # out = torch.sigmoid(out)

        amax = torch.argmax(attention)
        self.meta.update(f'{amax}')
        # print(self.meta.most_common(10))

        return out


class NeuralDictionaryV6Double(nn.Module):
    # Compares all keys with the query, computes the absolute differences per element between key and query, sums up the differences per key,
    #    then uses softmax to compute the probabilities per key and matrix multiplies the probabilities with the values.
    # Remove ununsed keys

    def __init__(self, in_features: int, out_features: int, capacity: int):
        # capacity,C, represents the number of key-value pairs.
        self.in_features = in_features
        self.out_features = out_features
        self.capacity = capacity
        super(NeuralDictionaryV6Double, self).__init__()
        # {C} keys each of size {in_features}, so the query needs to be of size {in_features}
        # for example 500 keys each of size 100, so the query needs to be of size 100
        self.keys = torch.zeros(1, in_features, dtype=torch.double)

        # {C} values each of size {out_features}, the output of the model will be of size {out_features}
        # for example 500 values each of size 4, the output of the model will be of size 4
        self.values = torch.zeros(1, out_features, dtype=torch.double)

        # to track and later see how many times a key has been chosen as the most important one(the key with the highest confidence)
        self.meta = Counter()

    def forward(self, query):
        # attention = torch.matmul(self.keys, query)
        query = torch.unsqueeze(query, 0)
        query = query.repeat(self.keys.shape[0], 1)  # now query has shape (500,100)
        attention = torch.abs(self.keys - query)  # computes absolute difference per element , (maybe later try euclidean distance or cosine similarity)
        attention = -torch.sum(attention,1)  # computes the sum of absolute differences per key, that is one key has one value, and makes it negative because of the following softmax operation
        attention = torch.softmax(attention, 0)  # compute the probabilities from the differences
        out = torch.matmul(attention, self.values)
        # use a activation function here if you want, like sigmoid, but that depends on the task, the output range we need
        # out = torch.sigmoid(out)

        amax = torch.argmax(attention)
        self.meta.update(f'{amax}')
        # print(self.meta.most_common(10))

        return out

    def update(self, key, value):
        key = torch.unsqueeze(key, 0)
        value = torch.unsqueeze(value, 0)
        self.keys = torch.cat((self.keys, key), 0)
        self.values = torch.cat((self.values, value), 0)

class NeuralDictionaryV6BDouble(nn.Module):
    # Compares all keys with the query, computes the absolute differences per element between key and query, sums up the differences per key,
    #    then uses softmax to compute the probabilities per key and matrix multiplies the probabilities with the values.
    # Remove ununsed keys

    def __init__(self, in_features: int, out_features: int, capacity: int):
        # capacity,C, represents the number of key-value pairs.
        self.in_features = in_features
        self.out_features = out_features
        super(NeuralDictionaryV6BDouble, self).__init__()
        # {C} keys each of size {in_features}, so the query needs to be of size {in_features}
        # for example 500 keys each of size 100, so the query needs to be of size 100
        self.keys = None

        # {C} values each of size {out_features}, the output of the model will be of size {out_features}
        # for example 500 values each of size 4, the output of the model will be of size 4
        self.values = None

        # to track and later see how many times a key has been chosen as the most important one(the key with the highest confidence)
        self.meta = Counter()

    def forward(self, query):
        # attention = torch.matmul(self.keys, query)
        # query = torch.unsqueeze(query, 0)
        # query = query.repeat(self.keys.shape[0], 1)  # now query has shape (500,100)
        attention = torch.abs(self.keys - query)  # computes absolute difference per element , (maybe later try euclidean distance or cosine similarity)
        attention = -torch.sum(attention,1)  # computes the sum of absolute differences per key, that is one key has one value, and makes it negative because of the following softmax operation
        attention = torch.softmax(attention, 0)  # compute the probabilities from the differences
        out = torch.matmul(attention, self.values)
        # use a activation function here if you want, like sigmoid, but that depends on the task, the output range we need
        out = torch.sigmoid(out)

        amax = torch.argmax(attention)
        # print(f'AMAX: {amax}')
        self.meta.update(f'{amax}')
        # print(self.meta.most_common(10))
        # return self.values[amax]
        return out

    def update(self, key, value):
        key = torch.unsqueeze(key, 0).float()
        value = torch.unsqueeze(value, 0).float()
        if self.keys is None:
            self.keys = key * 1.0
            self.values = value * 1.0
        else:
            self.keys = torch.cat((self.keys, key), 0)
            self.values = torch.cat((self.values, value), 0)


class NeuralDictionaryV7Double(nn.Module):
    # Compares all keys with the query, computes the absolute differences per element between key and query, sums up the differences per key,
    #    then uses softmax to compute the probabilities per key and matrix multiplies the probabilities with the values.
    # Remove ununsed keys

    def __init__(self, in_features: int, out_features: int, capacity: int):
        # capacity,C, represents the number of key-value pairs.
        self.in_features = in_features
        self.out_features = out_features
        self.capacity = capacity
        super(NeuralDictionaryV7Double, self).__init__()
        # {C} keys each of size {in_features}, so the query needs to be of size {in_features}
        # for example 500 keys each of size 100, so the query needs to be of size 100
        self.keys = [torch.zeros(in_features, dtype=torch.double)]

        # {C} values each of size {out_features}, the output of the model will be of size {out_features}
        # for example 500 values each of size 4, the output of the model will be of size 4
        self.values = [torch.zeros(out_features, dtype=torch.double)]

        # to track and later see how many times a key has been chosen as the most important one(the key with the highest confidence)
        self.meta = Counter()

    def forward(self, query):
        # attention = torch.matmul(self.keys, query)
        diffs = []
        for key in self.keys:
            diffs.append(-torch.sum(torch.abs(key - query)))

        attention = torch.softmax(torch.tensor(diffs), 0)  # compute the probabilities from the differences
        # out = torch.matmul(attention, torch.tensor(self.values))
        out = None
        for i in range(len(self.values)):
            if out is None:
                out = attention[i] * self.values[i]
            else:
                out += attention[i] * self.values[i]

        amax = torch.argmax(attention)
        self.meta.update(f'{amax}')
        # print(self.meta.most_common(10))

        return out

    def update(self, key, value):
        self.keys.append(key * 1)
        self.values.append(value * 1)


class NeuralDictionaryV8Double(nn.Module):
    # Compares all keys with the query, computes the absolute differences per element between key and query, sums up the differences per key,
    #    then uses softmax to compute the probabilities per key and matrix multiplies the probabilities with the values.
    # Remove ununsed keys

    def __init__(self, in_features: int, out_features: int, capacity: int):
        # capacity,C, represents the number of key-value pairs.
        self.in_features = in_features
        self.out_features = out_features
        self.capacity = capacity
        super(NeuralDictionaryV8Double, self).__init__()
        # {C} keys each of size {in_features}, so the query needs to be of size {in_features}
        # for example 500 keys each of size 100, so the query needs to be of size 100
        self.keys_ = torch.zeros(1, in_features, dtype=torch.double)
        self.keys = [torch.zeros(in_features, dtype=torch.double)]

        # {C} values each of size {out_features}, the output of the model will be of size {out_features}
        # for example 500 values each of size 4, the output of the model will be of size 4

        self.values_ = torch.zeros(1, out_features, dtype=torch.double)
        self.values = [torch.zeros(out_features, dtype=torch.double)]

        # to track and later see how many times a key has been chosen as the most important one(the key with the highest confidence)
        self.meta = Counter()
        self.flag = False

    def forward(self, query):
        if self.flag is True:
            self.rebuild()
            self.flag = False
        # attention = torch.matmul(self.keys, query)
        diffs = []
        for key in self.keys:
            diffs.append(-torch.sum(torch.abs(key - query)))

        attention = torch.softmax(torch.tensor(diffs), 0)  # compute the probabilities from the differences
        # out = torch.matmul(attention, torch.tensor(self.values))
        out = None
        for i in range(len(self.values)):
            if out is None:
                out = attention[i] * self.values[i]
            else:
                out += attention[i] * self.values[i]

        amax = torch.argmax(attention)
        self.meta.update(f'{amax}')
        # print(self.meta.most_common(10))

        return out

    def rebuild(self): # TODO: rebuild everytime after first forward after update
        self.values_ = None
        self.keys_ = None
        for i in range(len(self.keys)):
            key = torch.unsqueeze(self.keys[i], 0)
            value = torch.unsqueeze(self.values[i], 0)  # TODO: Fix this latter
            if self.keys_ is None and self.values_ is None:
                self.keys_ = key
                self.values_ = value
            else:
                self.keys_ = torch.cat((self.keys_, key), 0)
                self.values_ = torch.cat((self.values_, value), 0)

    def update(self, key, value):
        self.keys.append(key * 1)
        self.values.append(value * 1)
        self.rebuild()
        self.flag = True


# class NumberTable:
#     def __init__(self):
#         self.counter = 0
#         self.numbers = []
#     def get(self):
#         num = self.counter
#         self.numbers.append(num)
#         self.counter += 1
#         return num
#     def remove(self, num):
#         self.numbers.remove(num)
#     def check(self, num):
#         return num in self.numbers
#     def get_all(self):
#         return self.numbers

class NumberTable:
    def __init__(self, nums=1000):
        self.counter = 0
        self.table = [False for x in range(nums)]
    def get(self):
        for i in range(len(self.table)):
            s = self.table[i]
            if s is False:
                self.table[i] = True
                return i
        return None
    def remove(self, num):
        self.table[num] = False

    def check(self, num):
        if self.table[num] == True:
            return True
        else:
            False
        return None

from collections import namedtuple
# KeyValPair = namedtuple('KeyValPair', ['key', 'value', 'count'])
class KeyValPair:
    def __init__(self, key, value, count):
        self.key = key
        self.value = value
        self.count = count

class NeuralDictionaryV9Double(nn.Module):
    # Compares all keys with the query, computes the absolute differences per element between key and query, sums up the differences per key,
    #    then uses softmax to compute the probabilities per key and matrix multiplies the probabilities with the values.
    # Remove ununsed keys

    def __init__(self, in_features: int, out_features: int, capacity: int):
        # capacity,C, represents the number of key-value pairs.
        self.in_features = in_features
        self.out_features = out_features
        self.capacity = capacity
        super(NeuralDictionaryV9Double, self).__init__()
        # {C} keys each of size {in_features}, so the query needs to be of size {in_features}
        # for example 500 keys each of size 100, so the query needs to be of size 100
        self.keys = torch.zeros(1, in_features, dtype=torch.double)
        self.values = torch.zeros(1, out_features, dtype=torch.double)

        kvp = KeyValPair(
            key=torch.zeros(1, in_features, dtype=torch.double),
            value=torch.zeros(1, out_features, dtype=torch.double),
            count=0
        )
        self.numberTable = NumberTable(5000)
        self.dic = {self.numberTable.get(): kvp}
        self.table = [0]

        self.flag = False

    def forward(self, query):
        if self.flag is True:
            self.rebuild()
            self.flag = False

        query = torch.unsqueeze(query, 0)
        query = query.repeat(self.keys.shape[0], 1)  # now query has shape (500,100)
        attention = torch.abs(self.keys - query)  # computes absolute difference per element , (maybe later try euclidean distance or cosine similarity)
        attention = -torch.sum(attention,1)  # computes the sum of absolute differences per key, that is one key has one value, and makes it negative because of the following softmax operation
        attention = torch.softmax(attention, 0)  # compute the probabilities from the differences
        out = torch.matmul(attention, self.values)

        amax = torch.argmax(attention)
        self.dic[self.table[amax]].count += 1
        return out

    def rebuild(self): # TODO: rebuild everytime after first forward after update
        self.keys = None
        self.values = None

        self.table = []
        for k,kvp in self.dic.items():
            self.table.append(k)
            key = kvp.key
            value = kvp.value
            if self.keys is None and self.values is None:
                self.keys = key
                self.values = value
            else:
                # breakpoint()
                self.keys = torch.cat((self.keys, key), 0)
                self.values = torch.cat((self.values, value), 0)

        # for i in range(len(self.keys)):
        #     key = torch.unsqueeze(self.keys[i], 0)
        #     value = torch.unsqueeze(self.values[i], 0)  # TODO: Fix this latter
        #     if self.keys_ is None and self.values_ is None:
        #         self.keys_ = key
        #         self.values_ = value
        #     else:
        #         self.keys_ = torch.cat((self.keys_, key), 0)
        #         self.values_ = torch.cat((self.values_, value), 0)
    def trim(self):
        for k in list(self.dic.keys()):
            if self.dic[k].count == 0:
                del self.dic[k]
                self.numberTable.remove(k)

    def update(self, key, value):
        key = torch.unsqueeze(key, 0)
        value = torch.unsqueeze(value, 0)
        nr = self.numberTable.get()
        if nr is not None:
            kvp = KeyValPair(
                key=key,
                value=value,
                count=0
            )
            self.dic[nr] = kvp
        else:
            self.trim()
            nr = self.numberTable.get()
            if nr is not None:
                kvp = KeyValPair(
                    key=key,
                    value=value,
                    count=0
                )
                self.dic[nr] = kvp
        # self.keys.append(key * 1)
        # self.values.append(value * 1)
        # self.rebuild()
        self.flag = True


class NeuralDictionaryV5A(nn.Module):
    # Compares all keys with the query, computes the absolute differences per element between key and query, sums up the differences per key,
    #    then uses softmax to compute the probabilities per key and matrix multiplies the probabilities with the values.

    def __init__(self, in_features: int, out_features: int, capacity: int):
        # capacity,C, represents the number of key-value pairs.
        self.capacity = capacity
        super(NeuralDictionaryV5A, self).__init__()
        # {C} keys each of size {in_features}, so the query needs to be of size {in_features}
        # for example 500 keys each of size 100, so the query needs to be of size 100
        self.keys = nn.Parameter(torch.randn(capacity, in_features, dtype=torch.double))

        # {C} values each of size {out_features}, the output of the model will be of size {out_features}
        # for example 500 values each of size 4, the output of the model will be of size 4
        self.values = nn.Parameter(torch.randn(capacity, out_features, dtype=torch.double))

        self.multiplicator = nn.Parameter(torch.ones(in_features, dtype=torch.double))

        # to track and later see how many times a key has been chosen as the most important one(the key with the highest confidence)
        self.meta = Counter()

    def forward(self, query):
        # attention = torch.matmul(self.keys, query)
        query = self.multiplicator * query
        query = torch.unsqueeze(query, 0)
        query = query.repeat(self.capacity, 1)  # now query has shape (500,100)
        attention = torch.abs(self.keys - query)  # computes absolute difference per element , (maybe later try euclidean distance or cosine similarity)
        attention = -torch.sum(attention,1)  # computes the sum of absolute differences per key, that is one key has one value, and makes it negative because of the following softmax operation
        attention = torch.softmax(attention, 0)  # compute the probabilities from the differences
        out = torch.matmul(attention, self.values)
        # use a activation function here if you want, like sigmoid, but that depends on the task, the output range we need
        # out = torch.sigmoid(out)

        amax = torch.argmax(attention)
        self.meta.update(f'{amax}')
        # print(self.meta.most_common(10))

        return out


class NeuralDictionaryV5B(nn.Module):
    # Compares all keys with the query, computes the absolute differences per element between key and query, sums up the differences per key,
    #    then uses softmax to compute the probabilities per key and matrix multiplies the probabilities with the values.

    def __init__(self, in_features: int, out_features: int, capacity: int):
        # capacity,C, represents the number of key-value pairs.
        self.capacity = capacity
        super(NeuralDictionaryV5B, self).__init__()
        # {C} keys each of size {in_features}, so the query needs to be of size {in_features}
        # for example 500 keys each of size 100, so the query needs to be of size 100
        self.keys = nn.Parameter(torch.randn(capacity, in_features, dtype=torch.double))

        # {C} values each of size {out_features}, the output of the model will be of size {out_features}
        # for example 500 values each of size 4, the output of the model will be of size 4
        self.values = nn.Parameter(torch.randn(capacity, out_features, dtype=torch.double))

        self.multiplicator = nn.Parameter(torch.ones(in_features, dtype=torch.double))

        # to track and later see how many times a key has been chosen as the most important one(the key with the highest confidence)
        self.meta = Counter()

    def forward(self, query):
        query = torch.unsqueeze(query, 0)
        query = self.keys * query
        query = torch.sum(query, 0)
        query = query.repeat(self.capacity, 1)

        attention = torch.abs(self.keys - query)
        attention = -torch.sum(attention,1)  # computes the sum of absolute differences per key, that is one key has one value, and makes it negative because of the following softmax operation
        attention = torch.softmax(attention, 0)  # compute the probabilities from the differences
        out = torch.matmul(attention, self.values)
        # use a activation function here if you want, like sigmoid, but that depends on the task, the output range we need
        # out = torch.sigmoid(out)

        amax = torch.argmax(attention)
        self.meta.update(f'{amax}')
        # print(self.meta.most_common(10))

        return out

class NeuralDictionaryV5C(nn.Module):
    # Compares all keys with the query, computes the absolute differences per element between key and query, sums up the differences per key,
    #    then uses softmax to compute the probabilities per key and matrix multiplies the probabilities with the values.

    def __init__(self, in_features: int, out_features: int, capacity: int):
        # capacity,C, represents the number of key-value pairs.
        self.capacity = capacity
        super(NeuralDictionaryV5C, self).__init__()
        # {C} keys each of size {in_features}, so the query needs to be of size {in_features}
        # for example 500 keys each of size 100, so the query needs to be of size 100
        self.keys = nn.Parameter(torch.randn(capacity, in_features, dtype=torch.double))

        # {C} values each of size {out_features}, the output of the model will be of size {out_features}
        # for example 500 values each of size 4, the output of the model will be of size 4
        self.values = nn.Parameter(torch.randn(capacity, out_features, dtype=torch.double))

        self.multiplicator = nn.Parameter(torch.ones(in_features, dtype=torch.double))

        # to track and later see how many times a key has been chosen as the most important one(the key with the highest confidence)
        self.meta = Counter()

    def forward(self, query):
        query = torch.unsqueeze(query, 0)
        query = self.keys * query

        # want to use argmax
        # query = self.keys[]
        # query = torch.sum(query, 0)
        # query = query.repeat(self.capacity, 1)

        attention = torch.abs(self.keys - query)
        attention = -torch.sum(attention,1)  # computes the sum of absolute differences per key, that is one key has one value, and makes it negative because of the following softmax operation
        attention = torch.softmax(attention, 0)  # compute the probabilities from the differences
        out = torch.matmul(attention, self.values)
        # use a activation function here if you want, like sigmoid, but that depends on the task, the output range we need
        # out = torch.sigmoid(out)

        amax = torch.argmax(attention)
        self.meta.update(f'{amax}')
        # print(self.meta.most_common(10))

        return out

class NeuralDictionaryV5FDouble(nn.Module):
    # Compares all keys with the query, computes the absolute differences per element between key and query, sums up the differences per key,
    #    then uses softmax to compute the probabilities per key and matrix multiplies the probabilities with the values.

    def __init__(self, in_features: int, out_features: int, capacity: int):
        # capacity,C, represents the number of key-value pairs.
        self.capacity = capacity
        super(NeuralDictionaryV5FDouble, self).__init__()
        # {C} keys each of size {in_features}, so the query needs to be of size {in_features}
        # for example 500 keys each of size 100, so the query needs to be of size 100
        self.keys = nn.Parameter(torch.ones(capacity, in_features, dtype=torch.double), requires_grad=True)

        self.rand_attention = nn.Parameter(torch.ones(capacity, in_features, dtype=torch.double), requires_grad=True)

        # {C} values each of size {out_features}, the output of the model will be of size {out_features}
        # for example 500 values each of size 4, the output of the model will be of size 4
        self.values = nn.Parameter(torch.ones(capacity, out_features, dtype=torch.double), requires_grad=True)

        # to track and later see how many times a key has been chosen as the most important one(the key with the highest confidence)
        self.meta = Counter()

    def forward(self, query):
        # attention = torch.matmul(self.keys, query)
        query = torch.unsqueeze(query, 0)
        query = query.repeat(self.capacity, 1)  # now query has shape (500,100)
        attention = torch.abs((self.keys - query))
        # attention = torch.abs((self.keys + (self.rand_attention * torch.randn_like(self.rand_attention))) - query)  # computes absolute difference per element , (maybe later try euclidean distance or cosine similarity)
        attention = -torch.sum(attention,1)  # computes the sum of absolute differences per key, that is one key has one value, and makes it negative because of the following softmax operation
        attention = torch.softmax(attention, 0)  # compute the probabilities from the differences
        out = torch.matmul(attention, self.values)
        # use a activation function here if you want, like sigmoid, but that depends on the task, the output range we need
        # out = torch.sigmoid(out)

        amax = torch.argmax(attention)
        self.meta.update(f'{amax}')
        # print(self.meta.most_common(10))

        return out

class NeuralDictionaryV5GDouble(nn.Module):
    # Compares all keys with the query, computes the absolute differences per element between key and query, sums up the differences per key,
    #    then uses softmax to compute the probabilities per key and matrix multiplies the probabilities with the values.

    def __init__(self, in_features: int, out_features: int, capacity: int):
        # capacity,C, represents the number of key-value pairs.
        self.capacity = capacity
        super(NeuralDictionaryV5GDouble, self).__init__()
        # {C} keys each of size {in_features}, so the query needs to be of size {in_features}
        # for example 500 keys each of size 100, so the query needs to be of size 100
        self.keys = nn.Parameter(torch.ones(capacity, in_features, dtype=torch.double), requires_grad=True)

        # self.rand_attention = nn.Parameter(torch.ones(capacity, in_features, dtype=torch.double), requires_grad=True)

        # {C} values each of size {out_features}, the output of the model will be of size {out_features}
        # for example 500 values each of size 4, the output of the model will be of size 4
        self.values = nn.Parameter(torch.ones(capacity, out_features, dtype=torch.double), requires_grad=True)

        # to track and later see how many times a key has been chosen as the most important one(the key with the highest confidence)
        self.meta = Counter()

    def forward(self, query):
        # attention = torch.matmul(self.keys, query)
        query = torch.unsqueeze(query, 0)
        query = query.repeat(self.capacity, 1)  # now query has shape (500,100)
        attention = torch.abs((self.keys - query))
        # attention = torch.abs((self.keys + (self.rand_attention * torch.randn_like(self.rand_attention))) - query)  # computes absolute difference per element , (maybe later try euclidean distance or cosine similarity)
        attention = -torch.sum(attention,1)  # computes the sum of absolute differences per key, that is one key has one value, and makes it negative because of the following softmax operation
        attention = torch.softmax(attention, 0)  # compute the probabilities from the differences
        out = torch.matmul(attention, self.values)
        # use a activation function here if you want, like sigmoid, but that depends on the task, the output range we need
        # out = torch.sigmoid(out)

        amax = torch.argmax(attention)
        self.meta.update(f'{amax}')
        # print(self.meta.most_common(10))

        return out

class NeuralDictionaryV5GDouble(nn.Module):
    # Compares all keys with the query, computes the absolute differences per element between key and query, sums up the differences per key,
    #    then uses softmax to compute the probabilities per key and matrix multiplies the probabilities with the values.

    def __init__(self, in_features: int, out_features: int, capacity: int):
        # capacity,C, represents the number of key-value pairs.
        self.capacity = capacity
        super(NeuralDictionaryV5GDouble, self).__init__()
        # {C} keys each of size {in_features}, so the query needs to be of size {in_features}
        # for example 500 keys each of size 100, so the query needs to be of size 100
        self.keys = nn.Parameter(torch.rand(capacity, in_features, dtype=torch.double))

        # {C} values each of size {out_features}, the output of the model will be of size {out_features}
        # for example 500 values each of size 4, the output of the model will be of size 4
        self.values = nn.Parameter(torch.rand(capacity, out_features, dtype=torch.double))

        # to track and later see how many times a key has been chosen as the most important one(the key with the highest confidence)
        self.meta = Counter()

    def forward(self, query):
        # attention = torch.matmul(self.keys, query)
        query = torch.unsqueeze(query, 0)
        query = query.repeat(self.capacity, 1)  # now query has shape (500,100)
        attention = torch.abs(self.keys - query)  # computes absolute difference per element , (maybe later try euclidean distance or cosine similarity)
        attention = -torch.sum(attention,1)  # computes the sum of absolute differences per key, that is one key has one value, and makes it negative because of the following softmax operation
        attention = torch.softmax(attention, 0)  # compute the probabilities from the differences
        out = torch.matmul(attention, self.values)
        # use a activation function here if you want, like sigmoid, but that depends on the task, the output range we need
        # out = torch.sigmoid(out)

        amax = torch.argmax(attention)
        self.meta.update(f'{amax}')
        # print(self.meta.most_common(10))

        return out



class NeuralD2(nn.Module):
    def __init__(self, in_features: int, out_features: int, capacity: int, ):
        super(NeuralD2, self).__init__()
        self.l1 = NeuralDictionaryV5GDouble(in_features, 10, capacity)
        self.l2 = NeuralDictionaryV5GDouble(10, out_features, capacity)
    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        return x

class NeuralDictionaryV5PDouble(nn.Module):
    # Compares all keys with the query, computes the absolute differences per element between key and query, sums up the differences per key,
    #    then uses softmax to compute the probabilities per key and matrix multiplies the probabilities with the values.

    def __init__(self, in_features: int, out_features: int, capacity: int):
        # capacity,C, represents the number of key-value pairs.
        self.capacity = capacity
        super(NeuralDictionaryV5PDouble, self).__init__()
        # {C} keys each of size {in_features}, so the query needs to be of size {in_features}
        # for example 500 keys each of size 100, so the query needs to be of size 100
        self.keys = nn.Parameter(torch.ones(capacity, in_features, dtype=torch.double), requires_grad=True)

        # self.rand_attention = nn.Parameter(torch.ones(capacity, in_features, dtype=torch.double), requires_grad=True)

        # {C} values each of size {out_features}, the output of the model will be of size {out_features}
        # for example 500 values each of size 4, the output of the model will be of size 4
        self.values = nn.Parameter(torch.ones(capacity, out_features, dtype=torch.double), requires_grad=True)

        # to track and later see how many times a key has been chosen as the most important one(the key with the highest confidence)
        self.meta = Counter()

    def forward(self, query):
        # attention = torch.matmul(self.keys, query)
        query = torch.unsqueeze(query, 0)
        attention = torch.cosine_similarity(query, self.keys)
        # attention = torch.abs((self.keys + (self.rand_attention * torch.randn_like(self.rand_attention))) - query)  # computes absolute difference per element , (maybe later try euclidean distance or cosine similarity)
        attention = torch.softmax(attention, 0)  # compute the probabilities from the differences
        out = torch.matmul(attention, self.values)
        # use a activation function here if you want, like sigmoid, but that depends on the task, the output range we need
        # out = torch.sigmoid(out)

        amax = torch.argmax(attention)
        self.meta.update(f'{amax}')
        # print(self.meta.most_common(10))

        return out

class NeuralDictionaryV5RDouble(nn.Module):
    # Compares all keys with the query, computes the absolute differences per element between key and query, sums up the differences per key,
    #    then uses softmax to compute the probabilities per key and matrix multiplies the probabilities with the values.

    def __init__(self, in_features: int, out_features: int, capacity: int):
        # capacity,C, represents the number of key-value pairs.
        self.capacity = capacity
        super(NeuralDictionaryV5RDouble, self).__init__()
        # {C} keys each of size {in_features}, so the query needs to be of size {in_features}
        # for example 500 keys each of size 100, so the query needs to be of size 100
        self.keys = nn.Parameter(torch.randn(capacity, in_features, dtype=torch.double))

        # {C} values each of size {out_features}, the output of the model will be of size {out_features}
        # for example 500 values each of size 4, the output of the model will be of size 4
        self.values = nn.Parameter(torch.randn(capacity, out_features, dtype=torch.double))

        # to track and later see how many times a key has been chosen as the most important one(the key with the highest confidence)
        self.meta = Counter()

    def forward(self, query):
        # attention = torch.matmul(self.keys, query)
        query = torch.unsqueeze(query, 0)
        # query = query.repeat(self.capacity, 1)  # now query has shape (500,100)
        attention = torch.abs(self.keys - query)  # computes absolute difference per element , (maybe later try euclidean distance or cosine similarity)
        attention = -torch.sum(attention,1)  # computes the sum of absolute differences per key, that is one key has one value, and makes it negative because of the following softmax operation
        attention = torch.softmax(attention, 0)  # compute the probabilities from the differences
        out = torch.matmul(attention, self.values)
        # use a activation function here if you want, like sigmoid, but that depends on the task, the output range we need
        # out = torch.sigmoid(out)

        amax = torch.argmax(attention)
        self.meta.update(f'{amax}')
        # print(self.meta.most_common(10))

        return out

class NeuralDictionaryV5SDouble(nn.Module):
    # Compares all keys with the query, computes the absolute differences per element between key and query, sums up the differences per key,
    #    then uses softmax to compute the probabilities per key and matrix multiplies the probabilities with the values.

    def __init__(self, in_features: int, out_features: int, capacity: int):
        # capacity,C, represents the number of key-value pairs.
        self.capacity = capacity
        super(NeuralDictionaryV5SDouble, self).__init__()
        # {C} keys each of size {in_features}, so the query needs to be of size {in_features}
        # for example 500 keys each of size 100, so the query needs to be of size 100
        self.keys = nn.Parameter(torch.randn(capacity, in_features, dtype=torch.double))

        # {C} values each of size {out_features}, the output of the model will be of size {out_features}
        # for example 500 values each of size 4, the output of the model will be of size 4
        self.values = nn.Parameter(torch.randn(capacity, out_features, dtype=torch.double))

        self.keys2 = nn.Parameter(torch.randn(capacity, in_features, dtype=torch.double))
        self.values2 = nn.Parameter(torch.randn(capacity, out_features, dtype=torch.double))

        # to track and later see how many times a key has been chosen as the most important one(the key with the highest confidence)
        self.meta = Counter()

    def forward(self, query):
        # attention = torch.matmul(self.keys, query)
        query = torch.unsqueeze(query, 0)
        # query = query.repeat(self.capacity, 1)  # now query has shape (500,100)
        attention = torch.abs(self.keys - query)  # computes absolute difference per element , (maybe later try euclidean distance or cosine similarity)
        attention = -torch.sum(attention,1)  # computes the sum of absolute differences per key, that is one key has one value, and makes it negative because of the following softmax operation
        attention = torch.softmax(attention, 0)  # compute the probabilities from the differences
        # out = torch.matmul(attention, self.keys)
        out = torch.matmul(attention, self.keys)
        # use a activation function here if you want, like sigmoid, but that depends on the task, the output range we need
        # out = torch.sigmoid(out)

        attention = torch.abs(self.keys2 - out)
        attention = -torch.sum(attention, 1)
        attention = torch.softmax(attention, 0)
        out = torch.matmul(attention, self.values) # or use this? -> out = torch.matmul(attention, self.values)
        # print(out.shape)
        amax = torch.argmax(attention)
        self.meta.update(f'{amax}')
        # print(self.meta.most_common(10))

        return out


class NeuralD2(nn.Module):
    def __init__(self, in_features: int, out_features: int, capacity: int, ):
        super(NeuralD2, self).__init__()
        self.l1 = NeuralDictionaryV5GDouble(in_features, capacity, capacity)
        self.l2 = NeuralDictionaryV5GDouble(capacity, out_features, capacity)
    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        return x

class NeuralD3(nn.Module):
    def __init__(self, in_features: int, out_features: int, capacity: int, ):
        super(NeuralD3, self).__init__()
        self.l1 = NeuralDictionaryV5XDouble(in_features, capacity//2, capacity//4)
        self.l2 = NeuralDictionaryV5XDouble(capacity//2, out_features, capacity//4)
    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        return x

class Ann(nn.Module):
    def __init__(self, in_features: int, out_features: int, capacity: int, ):
        super(Ann, self).__init__()
        self.l1 = nn.Linear(in_features, 4096)
        self.l2 = nn.Linear(4096, 3)
        # self.l2 = nn.Linear(4096, out_features)
    def forward(self, x):
        # x = x.float()
        x = self.l1(x)
        x = torch.softmax(x, 0)
        x = self.l2(x)
        # x = self.l2(x)
        # x = self.l3(x)
        x = torch.sigmoid(x)
        return x


class NeuralDictionaryV10(nn.Module):
    # Compares all keys with the query, computes the absolute differences per element between key and query, sums up the differences per key,
    #    then uses softmax to compute the probabilities per key and matrix multiplies the probabilities with the values.
    # Remove ununsed keys

    def __init__(self, capacity: int):
        # capacity,C, represents the number of key-value pairs.
        self.capacity = capacity
        super(NeuralDictionaryV10, self).__init__()
        # {C} keys each of size {in_features}, so the query needs to be of size {in_features}
        # for example 500 keys each of size 100, so the query needs to be of size 100
        # self.keys = torch.zeros(1, in_features, dtype=torch.double)
        self.keys = nn.ParameterList()
        # {C} values each of size {out_features}, the output of the model will be of size {out_features}
        # for example 500 values each of size 4, the output of the model will be of size 4
        # self.values = torch.zeros(1, out_features, dtype=torch.double)
        self.values = None

        # to track and later see how many times a key has been chosen as the most important one(the key with the highest confidence)
        self.meta = Counter()

    def forward(self, query):
        # attention = torch.matmul(self.keys, query)
        attention = None
        for key in self.keys:
            # print(key.shape)
            # print(query.shape)
            at = -torch.abs(key - query).sum(1)
            if attention is None:
                attention = at * 1
            else:
                attention = torch.cat((attention, at))
        # attention = torch.abs(self.keys - query)  # computes absolute difference per element , (maybe later try euclidean distance or cosine similarity)
        # attention = -torch.sum(attention,1)  # computes the sum of absolute differences per key, that is one key has one value, and makes it negative because of the following softmax operation
        attention = torch.softmax(attention, 0)  # compute the probabilities from the differences
        out = torch.matmul(attention, self.values)
        # use a activation function here if you want, like sigmoid, but that depends on the task, the output range we need
        # out = torch.sigmoid(out)

        amax = torch.argmax(attention)
        self.meta.update(f'{amax}')
        # print(self.meta.most_common(10))

        return out

    def update(self, key, value):
        key = torch.unsqueeze(key, 0)
        value = torch.unsqueeze(value, 0)
        # self.keys = torch.cat((self.keys, key), 0)
        self.keys.append(nn.Parameter(key))
        if self.values is None:
            self.values = value * 1
        else:
            self.values = torch.cat((self.values, value), 0)

class NeuralDictionaryV11(nn.Module):
    # Compares all keys with the query, computes the absolute differences per element between key and query, sums up the differences per key,
    #    then uses softmax to compute the probabilities per key and matrix multiplies the probabilities with the values.
    # Remove ununsed keys

    def __init__(self, capacity: int):
        # capacity,C, represents the number of key-value pairs.
        self.capacity = capacity
        super(NeuralDictionaryV11, self).__init__()
        # {C} keys each of size {in_features}, so the query needs to be of size {in_features}
        # for example 500 keys each of size 100, so the query needs to be of size 100
        # self.keys = torch.zeros(1, in_features, dtype=torch.double)
        self.keys = None
        # {C} values each of size {out_features}, the output of the model will be of size {out_features}
        # for example 500 values each of size 4, the output of the model will be of size 4
        # self.values = torch.zeros(1, out_features, dtype=torch.double)
        self.values = None

        # to track and later see how many times a key has been chosen as the most important one(the key with the highest confidence)
        self.meta = Counter()

    def forward(self, query):
        # attention = torch.matmul(self.keys, query)
        attention = torch.abs(self.keys - query)  # computes absolute difference per element , (maybe later try euclidean distance or cosine similarity)
        attention = -torch.sum(attention,1)  # computes the sum of absolute differences per key, that is one key has one value, and makes it negative because of the following softmax operation
        attention = torch.softmax(attention, 0)
        # attention = torch.abs(self.keys - query)  # computes absolute difference per element , (maybe later try euclidean distance or cosine similarity)
        # attention = -torch.sum(attention,1)  # computes the sum of absolute differences per key, that is one key has one value, and makes it negative because of the following softmax operation
        attention = torch.softmax(attention, 0)  # compute the probabilities from the differences
        out = torch.matmul(attention, self.values)
        # use a activation function here if you want, like sigmoid, but that depends on the task, the output range we need
        # out = torch.sigmoid(out)

        amax = torch.argmax(attention)
        self.meta.update(f'{amax}')
        # print(self.meta.most_common(10))

        return out

    def update(self, key, value):
        key = torch.unsqueeze(key, 0)
        value = torch.unsqueeze(value, 0)
        # self.keys = torch.cat((self.keys, key), 0)
        # self.keys.append(nn.Parameter(key))
        if self.values is None:
            self.keys = nn.Parameter(key)
            self.values = value * 1
        else:
            self.keys = nn.Parameter(torch.cat((self.keys, key)))
            # print(self.keys.shape)
            self.values = torch.cat((self.values, value), 0)

class NeuralDictionaryV12(nn.Module):
    # Compares all keys with the query, computes the absolute differences per element between key and query, sums up the differences per key,
    #    then uses softmax to compute the probabilities per key and matrix multiplies the probabilities with the values.
    # Remove ununsed keys

    def __init__(self, capacity: int):
        # capacity,C, represents the number of key-value pairs.
        self.capacity = capacity
        super(NeuralDictionaryV12, self).__init__()
        # {C} keys each of size {in_features}, so the query needs to be of size {in_features}
        # for example 500 keys each of size 100, so the query needs to be of size 100
        # self.keys = torch.zeros(1, in_features, dtype=torch.double)
        self.keys = None
        # {C} values each of size {out_features}, the output of the model will be of size {out_features}
        # for example 500 values each of size 4, the output of the model will be of size 4
        # self.values = torch.zeros(1, out_features, dtype=torch.double)
        self.values = None

        # to track and later see how many times a key has been chosen as the most important one(the key with the highest confidence)
        self.meta = Counter()

    def forward(self, query):
        # attention = torch.matmul(self.keys, query)
        attention = torch.abs(self.keys - query)  # computes absolute difference per element , (maybe later try euclidean distance or cosine similarity)
        attention = -torch.sum(attention,1)  # computes the sum of absolute differences per key, that is one key has one value, and makes it negative because of the following softmax operation
        attention = torch.softmax(attention, 0)
        # attention = torch.abs(self.keys - query)  # computes absolute difference per element , (maybe later try euclidean distance or cosine similarity)
        # attention = -torch.sum(attention,1)  # computes the sum of absolute differences per key, that is one key has one value, and makes it negative because of the following softmax operation
        attention = torch.softmax(attention, 0)  # compute the probabilities from the differences
        mx = attention >= torch.max(attention)
        # print(mx)
        attention = attention * mx
        out = torch.matmul(attention, self.values)
        # use a activation function here if you want, like sigmoid, but that depends on the task, the output range we need
        out = torch.sigmoid(out)

        amax = torch.argmax(attention)
        self.meta.update(f'{amax}')
        # print(self.meta.most_common(10))

        return out

    def update(self, key, value):
        key = torch.unsqueeze(key, 0)
        value = torch.unsqueeze(value, 0)
        # self.keys = torch.cat((self.keys, key), 0)
        # self.keys.append(nn.Parameter(key))
        if self.values is None:
            self.keys = nn.Parameter(key)
            self.values = value * 1
        else:
            self.keys = nn.Parameter(torch.cat((self.keys, key)))
            # print(self.keys.shape)
            self.values = torch.cat((self.values, value), 0)

class NeuralDictionaryV13(nn.Module):
    # Compares all keys with the query, computes the absolute differences per element between key and query, sums up the differences per key,
    #    then uses softmax to compute the probabilities per key and matrix multiplies the probabilities with the values.
    # Remove ununsed keys

    def __init__(self, capacity: int):
        # capacity,C, represents the number of key-value pairs.
        self.capacity = capacity
        super(NeuralDictionaryV13, self).__init__()
        # {C} keys each of size {in_features}, so the query needs to be of size {in_features}
        # for example 500 keys each of size 100, so the query needs to be of size 100
        # self.keys = torch.zeros(1, in_features, dtype=torch.double)
        self.keys = None
        # {C} values each of size {out_features}, the output of the model will be of size {out_features}
        # for example 500 values each of size 4, the output of the model will be of size 4
        # self.values = torch.zeros(1, out_features, dtype=torch.double)
        self.values = None

        # to track and later see how many times a key has been chosen as the most important one(the key with the highest confidence)
        self.meta = Counter()

    def forward(self, query):
        attention = torch.matmul(self.keys, query)
        # attention = torch.abs(self.keys - query)  # computes absolute difference per element , (maybe later try euclidean distance or cosine similarity)
        # attention = -torch.sum(attention,1)  # computes the sum of absolute differences per key, that is one key has one value, and makes it negative because of the following softmax operation
        attention = torch.softmax(attention, 0)
        # attention = torch.abs(self.keys - query)  # computes absolute difference per element , (maybe later try euclidean distance or cosine similarity)
        # attention = -torch.sum(attention,1)  # computes the sum of absolute differences per key, that is one key has one value, and makes it negative because of the following softmax operation
        attention = torch.softmax(attention, 0)  # compute the probabilities from the differences
        mx = attention >= torch.max(attention)
        # print(mx)
        attention = attention * mx
        out = torch.matmul(attention, self.values)
        # use a activation function here if you want, like sigmoid, but that depends on the task, the output range we need
        out = torch.sigmoid(out)

        amax = torch.argmax(attention)
        self.meta.update(f'{amax}')
        # print(self.meta.most_common(10))

        return out

    def update(self, key, value):
        key = torch.unsqueeze(key, 0)
        value = torch.unsqueeze(value, 0)
        # self.keys = torch.cat((self.keys, key), 0)
        # self.keys.append(nn.Parameter(key))
        if self.values is None:
            self.keys = nn.Parameter(key)
            self.values = value * 1
        else:
            self.keys = nn.Parameter(torch.cat((self.keys, key)))
            # print(self.keys.shape)
            self.values = torch.cat((self.values, value), 0)

class NeuralDictionaryV14(nn.Module):
    # Compares all keys with the query, computes the absolute differences per element between key and query, sums up the differences per key,
    #    then uses softmax to compute the probabilities per key and matrix multiplies the probabilities with the values.
    # Remove ununsed keys

    def __init__(self, capacity: int):
        # capacity,C, represents the maximum number of key-value pairs.
        self.capacity = capacity
        super(NeuralDictionaryV14, self).__init__()
        # {C} keys each of size {in_features}, so the query needs to be of size {in_features}
        # for example 500 keys each of size 100, so the query needs to be of size 100
        # self.keys = torch.zeros(1, in_features, dtype=torch.double)
        self.keys = None
        # {C} values each of size {out_features}, the output of the model will be of size {out_features}
        # for example 500 values each of size 4, the output of the model will be of size 4
        # self.values = torch.zeros(1, out_features, dtype=torch.double)
        self.values = None

        # to track and later see how many times a key has been chosen as the most important one(the key with the highest confidence)
        self.meta = Counter()

    def forward(self, query):
        attention = torch.matmul(self.keys, query)
        # attention = torch.abs(self.keys - query)  # computes absolute difference per element , (maybe later try euclidean distance or cosine similarity)
        # attention = -torch.sum(attention,1)  # computes the sum of absolute differences per key, that is one key has one value, and makes it negative because of the following softmax operation
        attention = torch.softmax(attention, 0)
        # attention = torch.abs(self.keys - query)  # computes absolute difference per element , (maybe later try euclidean distance or cosine similarity)
        # attention = -torch.sum(attention,1)  # computes the sum of absolute differences per key, that is one key has one value, and makes it negative because of the following softmax operation
        attention = torch.softmax(attention, 0)  # compute the probabilities from the differences
        mx = attention >= torch.max(attention)
        # print(mx)
        attention = attention * mx
        out = torch.matmul(attention, self.values)
        # use a activation function here if you want, like sigmoid, but that depends on the task, the output range we need
        # out = torch.sigmoid(out)

        amax = torch.argmax(attention)
        self.meta.update(f'{amax}')
        # print(self.meta.most_common(10))

        return out

    def update(self, key, value):
        key = torch.unsqueeze(key, 0)
        value = torch.unsqueeze(value, 0)
        # self.keys = torch.cat((self.keys, key), 0)
        # self.keys.append(nn.Parameter(key))
        if self.values is None:
            self.keys = nn.Parameter(key)
            self.values = value * 1
        else:
            self.keys = nn.Parameter(torch.cat((self.keys, key)))
            # print(self.keys.shape)
            self.values = torch.cat((self.values, value), 0)

class NeuralDictionaryV15(nn.Module):

    def __init__(self, capacity: int):
        super(NeuralDictionaryV15, self).__init__()
        # capacity represents the maximum number of key-value pairs.
        self.capacity = capacity
        self.keys = None
        self.values = None

        # to track and later see how many times a key has been chosen as the most important one(the key with the highest confidence)
        self.meta = Counter()

    def forward(self, query):
        attention = torch.matmul(self.keys, query)
        attention = torch.softmax(attention, 0)
        attention = (attention >= torch.max(attention)) * attention
        out = torch.matmul(attention, self.values)
        # use a activation function here if you want, like sigmoid, but that depends on the task, the output range we need
        # out = torch.sigmoid(out)

        amax = torch.argmax(attention)
        self.meta.update(f'{amax}')
        # print(self.meta.most_common(10))

        return out

    def update(self, key, value):
        key = torch.unsqueeze(key, 0)
        value = torch.unsqueeze(value, 0)
        if self.values is None:
            self.keys = nn.Parameter(key)
            self.values = value * 1
        else:
            self.keys = nn.Parameter(torch.cat((self.keys, key)))
            # print(self.keys.shape)
            self.values = torch.cat((self.values, value), 0)

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super(Linear, self).__init__()
        self.weights = nn.Parameter(torch.rand(out_features, in_features, dtype=torch.double))

    def forward(self, x):
        # x = torch.matmul(x, self.weights)
        x = torch.matmul(self.weights, x)
        return x


class NeuralDictionaryNext(nn.Module):
    # Compares all keys with the query, computes the absolute differences per element between key and query, sums up the differences per key,
    #    then uses softmax to compute the probabilities per key and matrix multiplies the probabilities with the values.

    def __init__(self, in_features: int, out_features: int, capacity: int):
        # capacity,C, represents the number of key-value pairs.
        self.capacity = capacity
        super(NeuralDictionaryNext, self).__init__()
        # {C} keys each of size {in_features}, so the query needs to be of size {in_features}
        # for example 500 keys each of size 100, so the query needs to be of size 100
        self.keys = nn.Parameter(torch.randn(capacity, in_features, dtype=torch.double), requires_grad=True)

        # {C} values each of size {in_features}, the output of the model will be of size {in_features}
        # for example 500 values each of size 100, the output of the model will be of size 100
        self.values = nn.Parameter(torch.randn(capacity, in_features, dtype=torch.double), requires_grad=True)

        # self.linear = Linear(in_features=in_features, out_features=out_features)
        self.linear = nn.Linear(in_features=in_features, out_features=out_features)
        # self.keys2 = nn.Parameter(torch.randn(capacity, in_features, dtype=torch.double), requires_grad=True)
        # self.values2 = nn.Parameter(torch.randn(capacity, out_features, dtype=torch.double), requires_grad=True)
        # self.layer = nn.Linear(in_features=in_features, out_features=out_features)
        # self.values = nn.Parameter(torch.randn(capacity, out_features, dtype=torch.double))

        # to track and later see how many times a key has been chosen as the most important one(the key with the highest confidence)
        self.meta = Counter()

    def forward(self, query):
        attention = torch.abs(self.keys - query)  # computes absolute difference per element , (maybe later try euclidean distance or cosine similarity)
        attention = -torch.sum(attention,1)  # computes the sum of absolute differences per key, that is one key has one value, and makes it negative because of the following softmax operation
        attention = torch.softmax(attention, 0)  # compute the probabilities from the differences

        amax = torch.argmax(attention)
        self.meta.update(f'{amax}')
        # print(self.meta.most_common(10))

        attention = torch.unsqueeze(attention, 1)
        out = attention * self.values
        out = out.sum(0)
        # print(out.shape)

        out = torch.tensor(out, dtype=torch.double)
        out = self.linear(out)
        # print(out.shape)
        # exit()
        # attention = torch.abs(self.keys2 - out)  # computes absolute difference per element , (maybe later try euclidean distance or cosine similarity)
        # attention = -torch.sum(attention,1)  # computes the sum of absolute differences per key, that is one key has one value, and makes it negative because of the following softmax operation
        # attention = torch.softmax(attention, 0)
        # out = torch.matmul(attention, self.values2)
        # use a activation function here if you want, like sigmoid, but that depends on the task, the output range we need
        # out = torch.sigmoid(out)

        return out

class NeuralDictionaryNext2(nn.Module):
    # Compares all keys with the query, computes the absolute differences per element between key and query, sums up the differences per key,
    #    then uses softmax to compute the probabilities per key and matrix multiplies the probabilities with the values.

    def __init__(self, in_features: int, out_features: int, capacity: int):
        # capacity,C, represents the number of key-value pairs.
        self.capacity = capacity
        super(NeuralDictionaryNext2, self).__init__()
        # {C} keys each of size {in_features}, so the query needs to be of size {in_features}
        # for example 500 keys each of size 100, so the query needs to be of size 100
        self.keys = nn.Parameter(torch.randn(capacity, in_features, dtype=torch.double), requires_grad=True)

        # {C} values each of size {in_features}, the output of the model will be of size {in_features}
        # for example 500 values each of size 100, the output of the model will be of size 100
        self.values = nn.Parameter(torch.randn(capacity, in_features, dtype=torch.double), requires_grad=True)

        # self.linear = nn.Linear(in_features=in_features, out_features=out_features)
        self.linear = Linear(in_features=in_features, out_features=out_features)

        # to track and later see how many times a key has been chosen as the most important one(the key with the highest confidence)
        self.meta = Counter()

    def forward(self, query):
        attention = torch.abs(self.keys - query)  # computes absolute difference per element , (maybe later try euclidean distance or cosine similarity)
        attention = -torch.sum(attention,1)  # computes the sum of absolute differences per key, that is one key has one value, and makes it negative because of the following softmax operation
        attention = torch.softmax(attention, 0)  # compute the probabilities from the differences

        amax = torch.argmax(attention)
        self.meta.update(f'{amax}')
        # print(self.meta.most_common(10))

        attention = torch.unsqueeze(attention, 1)
        out = attention * self.values
        out = out.sum(0)

        out = torch.tensor(out, dtype=torch.double)
        out = self.linear(out)
        # attention = torch.abs(self.keys2 - out)  # computes absolute difference per element , (maybe later try euclidean distance or cosine similarity)
        # attention = -torch.sum(attention,1)  # computes the sum of absolute differences per key, that is one key has one value, and makes it negative because of the following softmax operation
        # attention = torch.softmax(attention, 0)
        # out = torch.matmul(attention, self.values2)
        # use a activation function here if you want, like sigmoid, but that depends on the task, the output range we need
        # out = torch.sigmoid(out)

        return out


class NeuralDictionaryNext3(nn.Module):
    # Compares all keys with the query, computes the absolute differences per element between key and query, sums up the differences per key,
    #    then uses softmax to compute the probabilities per key and matrix multiplies the probabilities with the values.

    def __init__(self, in_features: int, out_features: int, capacity: int):
        super(NeuralDictionaryNext3, self).__init__()
        # capacity,C, represents the number of key-value pairs.
        self.capacity = capacity
        # {C} keys each of size {in_features}, so the query needs to be of size {in_features}
        # for example 500 keys each of size 100, so the query needs to be of size 100
        self.keys = nn.Parameter(torch.randn(capacity, in_features, dtype=torch.double), requires_grad=True)

        # {C} values each of size {in_features}, the output of the model will be of size {in_features}
        # for example 500 values each of size 100, the output of the model will be of size 100
        self.values = nn.Parameter(torch.randn(capacity, in_features, dtype=torch.double), requires_grad=True)

        # self.linear = nn.Linear(in_features=in_features, out_features=out_features)
        self.linear = Linear(in_features=in_features, out_features=out_features)

        # to track and later see how many times a key has been chosen as the most important one(the key with the highest confidence)
        self.meta = Counter()

    def forward(self, query):
        attention = torch.abs(self.keys - query)  # computes absolute difference per element , (maybe later try euclidean distance or cosine similarity)
        attention = -torch.sum(attention,1)  # computes the sum of absolute differences per key, that is one key has one value, and makes it negative because of the following softmax operation

        amax = torch.argmax(attention)
        self.meta.update(f'{amax}')
        # print(self.meta.most_common(10))

        attention = torch.unsqueeze(attention, 1)
        out = attention * self.values
        out = out.sum(0)

        out = torch.tensor(out, dtype=torch.double)
        out = self.linear(out)

        return out


# from torch.optim import Adam, SGD
# criterion = torch.nn.MSELoss()
# # net = NeuralDictionaryV5SDouble(in_features=3, out_features=1, capacity=100)
# # net = NeuralDictionaryV10(100)
# net = NeuralDictionaryNext2(in_features=3, out_features=1, capacity=5)
# l = []
# l.append((torch.tensor([1,0,1], dtype=float), torch.tensor([1], dtype=float)))
# l.append((torch.tensor([1,1,1], dtype=float), torch.tensor([2], dtype=float)))
# l.append((torch.tensor([1,0.9,1], dtype=float), torch.tensor([1], dtype=float)))
# for x in range(10):
#     l.append(
#         (torch.randn(3, dtype=torch.double), torch.randn(1, dtype=torch.double))
#     )
# for key, val in l:
#     net.update(key, val)
#
# optim = Adam(net.parameters(), lr=0.0005)
# # optim = SGD(net.parameters(), lr=0.005)
# flag = False
# while True:
#     optim.zero_grad()
#     loss_ = torch.tensor([0], dtype=float)
#     s = ''
#     for x, y in l:
#         o = net(x.detach())
#         # loss += criterion(o, y.detach())
#         loss_ = torch.cat((loss_, criterion(o, y.detach()).unsqueeze(0)))
#         s = s + f"Y: {y} O: {o}"
#     print(s)
#     loss = torch.median(loss_)
#     print(f'LOSS: {loss}')
#     if loss < 0.001:
#         # print(net.keys)
#         # print(net.rand_attention)
#         break
#     loss.backward()
#     # loss.backward(retain_graph=True)
#     optim.step()
#     print([key[0] for key in net.keys])

from torch.optim import Adam, SGD
criterion = torch.nn.MSELoss()
# net = NeuralDictionaryV5SDouble(in_features=3, out_features=1, capacity=100)
# net = NeuralDictionaryV10(100)
net = NeuralDictionaryNext2(in_features=3, out_features=1, capacity=100)
l = []
l.append((torch.tensor([1,0,1], dtype=float), torch.tensor([1], dtype=float)))
l.append((torch.tensor([1,1,1], dtype=float), torch.tensor([2], dtype=float)))
l.append((torch.tensor([1,0.9,1], dtype=float), torch.tensor([1], dtype=float)))
for x in range(10):
    l.append(
        (torch.randn(3, dtype=torch.double), torch.randn(1, dtype=torch.double))
    )
# for key, val in l:
#     net.update(key, val)
#
# optim = Adam(net.parameters(), lr=0.0005)
optim = SGD(net.parameters(), lr=0.05)
flag = False
while True:
    optim.zero_grad()
    loss_ = torch.tensor([0], dtype=float)
    s = ''
    for x, y in l:
        o = net(x.detach())
        # loss += criterion(o, y.detach())
        loss_ = torch.cat((loss_, criterion(o, y.detach()).unsqueeze(0)))
        s = s + f"Y: {y} O: {o}"
    print(s)
    loss = torch.median(loss_)
    print(f'LOSS: {loss}')
    if loss < 0.001:
        # print(net.keys)
        # print(net.rand_attention)
        break
    loss.backward()
    # loss.backward(retain_graph=True)
    optim.step()
    print([key[0] for key in net.keys])