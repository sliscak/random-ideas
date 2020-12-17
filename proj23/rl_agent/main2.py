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
# from new_datasets import WikiDataset
from collections import Counter, namedtuple, deque
import model

# def sort_fun():
def learn(replay_memory):
    pass



import gym
env = gym.make('BipedalWalker-v3')
#env.action_space       Box(-1.0, 1.0, (4,), float32)
#env.observation_space  Box(-inf, inf, (24,), float32)
# breakpoint()
agent = model.Net()
criterion = nn.MSELoss()
optimizer = optim.AdamW(agent.parameters(), lr=0.05)
# replay_memory = deque(maxlen=10)
replay_memory = []

while True:
    for i_episode in range(5):
        observation = env.reset()
        eps_reward = 0
        replay = deque(maxlen=100)
        flag = False
        for t in range(100):
            env.render()
            # print(observation)
            # action = env.action_space.sample()
            if flag:
                action = agent(torch.tensor(observation, dtype=torch.double)).detach().numpy()
            else:
                action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            eps_reward += reward
            # print(f'Reward: {reward}')
            replay.append((observation, action))
            if done:
                break
        flag = not flag
        print("Episode finished after {} timesteps".format(t + 1))
        print(f'EP:{i_episode}\tREWARD: {eps_reward}')
        # print(replay)
        replay_memory.append((list(replay), eps_reward))
    replay_memory.sort(key=lambda array: array[1], reverse=True)
    loss = torch.tensor([0], dtype=torch.double)
    # replay = replay_memory[0][0]
    for observation, action in replay_memory[0][0]:
        out = agent(torch.tensor(observation))
        loss += criterion(out, torch.tensor(action, dtype=torch.double))
        # print(f'LOSS: {loss}')
        # exit()
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        new_loss = torch.tensor([0], dtype=torch.double)
        # replay = replay_memory[0][0]
        for observation, action in replay_memory[0][0]:
            out = agent(torch.tensor(observation))
            new_loss += criterion(out, torch.tensor(action, dtype=torch.double))
    print(f'OLD LOSS: {loss.detach()}\nNEW LOSS: {new_loss}')
    replay_memory = []
    # print([x[1] for x in replay_memory])
    # learn(replay_memory)
    # sleep(4)
env.close()