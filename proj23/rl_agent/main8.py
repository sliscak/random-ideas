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
env = gym.make('BipedalWalker-v3')
#env.action_space       Box(-1.0, 1.0, (4,), float32)
#env.observation_space  Box(-inf, inf, (24,), float32)
# breakpoint()
learning = True
agent = model.Net()
rewardNet = model.Net2()
criterion = nn.MSELoss()
optimizer = optim.AdamW(agent.parameters(), lr=0.005)
rewardNet_opt = optim.AdamW(rewardNet.parameters(), lr=0.005)
# replay_memory = deque(maxlen=10)
replay_memory = deque(maxlen=2)

while True:
    learning = True
    for i_episode in range(1):
        observation = env.reset()
        eps_reward = 0
        replay = deque(maxlen=1000)
        flag = True
        for t in range(1000):
            env.render()
            # print(observation)
            # action = env.action_space.sample()
            if flag:
                # PROBE/STOCHASTICALI SELECT FROM RANDOM ACTIONS THE ONE WITH HIGHEST EXPECTED FURURE REWARD
                with torch.no_grad():
                    rand_actions = (np.random.random((10, 4)) * 2) - 1
                    rews = []
                    for action in rand_actions:
                        flat = np.concatenate((observation, action))
                        rew = rewardNet(torch.tensor(flat, dtype=torch.double))
                        rews.append((action, rew))
                    rews.sort(key=lambda array: array[1], reverse=True)
                    action = rews[0][0]
                    # print(action)
                    # exit()
                # action = agent(torch.tensor(observation, dtype=torch.double)).detach().numpy()
            else:
                action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            eps_reward += reward
            # print(f'Reward: {reward}')
            replay.append((observation, action))
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
        for replay in replay_memory:
            global_reward = replay[1]
            # if global_reward < 0:
            #     continue
            rewardNet_opt.zero_grad()
            pred_rew = torch.tensor([0], dtype=torch.double)
            for observation, action in replay[0]:
                flat = np.concatenate((observation, action))
                pred_rew += rewardNet(torch.tensor(flat, dtype=torch.double))
                # print('JERE')
                # exit()
                # loss += criterion(out, torch.tensor(action, dtype=torch.double))
                # print(f'LOSS: {loss}')
                # exit()
            loss = criterion(pred_rew, torch.tensor([replay[1]], dtype=torch.double))
            loss.backward()
            rewardNet_opt.step()

            # --------------------------------------
            with torch.no_grad():
                new_pred_rew = torch.tensor([0], dtype=torch.double)
                # replay = replay_memory[0][0]
                for observation, action in replay_memory[0][0]:
                    flat = np.concatenate((observation, action))
                    new_pred_rew += rewardNet(torch.tensor(flat, dtype=torch.double))
                new_loss = criterion(new_pred_rew, torch.tensor([replay_memory[0][1]], dtype=torch.double))
        #
                print(f'PRED LOSS: {loss.detach()}\nNEW LOSS: {new_loss}')
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