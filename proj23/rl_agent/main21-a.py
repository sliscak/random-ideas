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
from tqdm import tqdm
env = gym.make('BipedalWalker-v3')
# env = gym.make('CarRacing-v0')
#env.action_space       Box(-1.0, 1.0, (4,), float32)
#env.observation_space  Box(-inf, inf, (24,), float32)
print(env.action_space)
print(env.observation_space)
# exit()
# breakpoint()
learning = True
criterion = nn.MSELoss()

# agent = model.Net()
# optimizer = optim.AdamW(agent.parameters(), lr=0.05)
#
# rewardNet = model.NeuralDictionaryV4()
# rewardNet_opt = optim.AdamW(rewardNet.parameters(), lr=0.05)

# oracle = model.NeuralDictionaryV5A(in_features=76, out_features=24, capacity=500)
oracle = model.NeuralD2(in_features=76, out_features=24, capacity=500)
oracle_opt = optim.AdamW(oracle.parameters(), lr=0.005)
# replay_memory = deque(maxlen=10)
replay_memory = deque(maxlen=10)

Capture = namedtuple('Capture', ['observation', 'action', 'new_observation', 'reward', 'global_reward'])
# env.
while True:
    learning = True
    for i_episode in range(2):
        observation = env.reset()
        # observation = observation.flatten()
        eps_reward = 0
        replay = deque(maxlen=1000)
        obs = deque([observation for x in range(3)], maxlen=3)
        flag = True
        pred_diffs = []
        for t in range(500):
            env.render()

            action = env.action_space.sample()
            np_obs = np.array(obs).flatten()
            flat = np.concatenate((np_obs, action))
            with torch.no_grad():
                future_state = oracle(torch.tensor(flat, dtype=torch.double)).numpy()

            observation, reward, done, info = env.step(action)
            observation = observation.flatten()
            old_obs = np.array(obs).flatten()
            obs.append(observation)
            new_obs = np.array(observation).flatten()
            eps_reward += reward
            cap = Capture(observation=old_obs, action=action, new_observation=new_obs, reward=reward, global_reward=eps_reward)
            pred_diffs.append( torch.sum(torch.abs(torch.tensor(future_state) - torch.tensor(new_obs))))
            # oracle.zero_grad()
            # loss = torch.tensor([0], dtype=torch.double)
            # flat = np.concatenate((cap.observation, cap.action))
            # out = oracle(torch.tensor(flat, dtype=torch.double))
            # loss = criterion(out, torch.tensor(cap.new_observation, dtype=torch.double))
            # loss.backward()
            # oracle_opt.step()
            # # --------------------------------------
            # with torch.no_grad():
            #     new_loss = torch.tensor([0], dtype=torch.double)
            #     flat = np.concatenate((cap.observation, cap.action))
            #     out = oracle(torch.tensor(flat, dtype=torch.double))
            #     new_loss = criterion(out, torch.tensor(cap.new_observation, dtype=torch.double))
            #     print(f'PRED LOSS: {loss.detach()}\nNEW LOSS: {new_loss}')

            replay.append(cap)
            if done:
                break
        # flag = not flag
        median_diff = np.median(np.array(pred_diffs))
        print(f'MEDIAN PREDICTION DIFF: {median_diff}')

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
    # while True:
    for x in tqdm(range(10)):
    # if learning:
        sflag = False
        for replay in replay_memory:
            if sflag is False:
                oracle.zero_grad()
                loss = torch.tensor([0], dtype=torch.double)
                for cap in replay[0]:
                    flat = np.concatenate((cap.observation, cap.action))
                    out = oracle(torch.tensor(flat, dtype=torch.double))
                    loss += criterion(out, torch.tensor(cap.new_observation, dtype=torch.double))
                loss.backward()
                sflag = True
            oracle_opt.step()
            # --------------------------------------
            # with torch.no_grad():
            #     new_loss = torch.tensor([0], dtype=torch.double)
            #     for cap in replay[0]:
            #         flat = np.concatenate((cap.observation, cap.action))
            #         out = oracle(torch.tensor(flat, dtype=torch.double))
            #         new_loss += criterion(out, torch.tensor(cap.new_observation, dtype=torch.double))
            # print(f'PRED LOSS: {loss.detach()}\nNEW LOSS: {new_loss}')

env.close()