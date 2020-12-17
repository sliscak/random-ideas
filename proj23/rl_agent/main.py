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
# from random import random

# def sort_fun():
def learn(replay_memory):
    pass



import gym
# env = gym.make('BipedalWalker-v3')
# import gym
env = gym.make("procgen:procgen-coinrun-v0", start_level=0, num_levels=1)
# env = gym.make('CarRacing-v0')
#env.action_space       Box(-1.0, 1.0, (4,), float32)
#env.observation_space  Box(-inf, inf, (24,), float32)
print(env.action_space)
print(env.observation_space)
# exit()
# breakpoint()
learning = True
agent = model.Net()
rewardNet = model.Net2()
mimic = model.Net3()

criterion = nn.MSELoss()

optimizer = optim.AdamW(agent.parameters(), lr=0.05)
rewardNet_opt = optim.AdamW(rewardNet.parameters(), lr=0.05)
mimic_opt = optim.AdamW(mimic.parameters(), lr=0.05)

replay_memory = deque(maxlen=2)

st_col = st.beta_columns(2)
# for col in st_col:
#     col.empty()
st_col[0] = st.empty()
st_col[1] = st.empty()
while True:
    learning = True
    for i_episode in range(1):
        observation = env.reset()
        # observation = observation.flatten()
        eps_reward = 0
        replay = deque(maxlen=1000)
        obs = deque([observation/255 for x in range(3)], maxlen=3)
        flag = True
        for t in range(100):
            action = env.action_space.sample()
            action_array = np.zeros(15)
            action_array[action] = 1
            st_col[0].image(image=observation,caption='GAME', width=250)
            observation, reward, done, info = env.step(action)
            observation = observation / 255
            obs.append(observation)
            np_obs = np.array(obs).flatten()
            with torch.no_grad():
                flat = np.concatenate((np_obs, action_array)).flatten()
                mim_img = mimic(torch.tensor(flat, dtype=torch.double))
                mim_img = torch.reshape(mim_img[0:12288], (64, 64, 3)).numpy()
                st_col[1].image(image=mim_img, caption='MIMIC', width=250)
            eps_reward += reward
            # print(f'Reward: {reward}')
            replay.append((np_obs, action_array))
            if done:
                break
        # flag = not flag
        print("Episode finished after {} timesteps".format(t + 1))
        print(f'EP:{i_episode}\tREWARD: {eps_reward}')
        # print(replay)
        replay_memory.append((list(replay), eps_reward))
    # loss = torch.tensor([0], dtype=torch.double)
    # replay = replay_memory[0][0]
    # while True:
    #     for replay in replay_memory:
    #         global_reward = replay[1]
    #         for observation, action in replay[0]:
    #             flat = np.concatenate((observation, action))
    #             pred_rew += rewardNet(torch.tensor(flat, dtype=torch.double))
    # exit()
    # --------------------------------------------------
    # while learning:

    if learning:
        st.write('learning')
        for replay in replay_memory:
            mimic_opt.zero_grad()

            # loss = torch.tensor([0], dtype=torch.double)
            for observation, action in replay[0]:
                flat = np.concatenate((observation, action)).flatten()
                out = mimic(torch.tensor(flat, dtype=torch.double))
                loss = torch.tensor([0], dtype=torch.double)
                loss += criterion(out, torch.tensor(flat, dtype=torch.double))
                loss.backward()
                mimic_opt.step()
            # exit()
            # --------------------------------------
            with torch.no_grad():
                new_pred_loss = torch.tensor([0], dtype=torch.double)
                for observation, action in replay[0]:
                    flat = np.concatenate((observation, action)).flatten()
                    out = mimic(torch.tensor(flat, dtype=torch.double))
                    new_pred_loss += criterion(out, torch.tensor(flat, dtype=torch.double))
                print(f'PRED LOSS: {loss.detach()}\nNEW LOSS: {new_pred_loss}')
            # if new_loss < 0.02:
            #     learning = False
    # print(f'PRED REW: {pred_rew.detach()}\nNEW_PRED: {new_pred_rew}')
    # optimizer.step()
    # with torch.no_grad():
    #     new_loss = torch.tensor([0], dtype=torch.double)
    #     # replay = replay_memory[0][0]
    #     for observation, action in replay_memory[0][0]:
    #         out = agent(torch.tensor(observation))
    #         new_loss += criterion(out, torch.tensor(action, dtype=torch.double))
    # print(f'OLD LOSS: {loss.detach()}\nNEW LOSS: {new_loss}')
    # replay_memory = []
    # print([x[1] for x in replay_memory])
    # learn(replay_memory)
    # sleep(4)
env.close()