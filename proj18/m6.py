import gym
import numpy as np
from collections import namedtuple
# import torch
#from torch.nn import modules as nn
# from torch import nn
from collections import deque

''''
    INPUT: 10
    MEM_STATE: 13
    OUTPUT: 3
'''
# The Puppet
# class GameNet(nn.Module):
#     def __init__(self):
#         super(GameNet, self).__init__()
#         self.l1 = nn.Sequential(
#             nn.Linear(13, 20),
#             nn.ReLU(),
#             nn.Linear(20, 10),
#             nn.Tanh(),
#         )
#         self.l2 = nn.Sequential(
#             nn.Linear(10, 20),
#             nn.ReLU(),
#             nn.Linear(20, 20),
#             nn.ReLU(),
#             nn.Linear(20, 1),
#             nn.Tanh(),
#         )
#         self.l3 = nn.Sequential(
#             nn.Linear(1, 1),
#             nn.Sigmoid(),
#         )
#
#     def forward(self, input, mem_state):
#         x1 = self.l2(input.float())
#
#         x2 = self.l1(mem_state.float())
#         x2 = self.l2(x2)
#         print((x1, x2))
#         x3 = torch.abs(x1) - torch.abs(x2)
#         #x3 = x3.unsqueeze(0)
#         return x3

# gamenet = GameNet()
# optimizer = torch.optim.Adam(params=gamenet.parameters(), lr=0.001)
# criterion = torch.nn.MSELoss()

def search(old_observation, action, observation, memory):
    scores = []
    for m in list(memory)[:len(memory)//2]:
        score1 = np.sum((m.old_obs - old_observation) ** 2)
        score2 = np.sum((m.action - action) ** 2)
        score3 = np.sum((m.obs - observation) ** 2)
        score = score1 + score2 + score3
        scores.append(Hit(score1, score2, score3, score, m))
    scores.sort(key=lambda hit: hit.old_obs_diff + hit.action_diff)
    # print(f'Found: {scores[0]}\nQuery: {(old_observation, action, observation)}')
    return scores[0]


Hit = namedtuple('Hit', 'old_obs_diff action_diff obs_diff sum_diff mem')
Mem = namedtuple('Mem', 'old_obs action obs')

env = gym.make('BipedalWalker-v3')
memory = deque(maxlen=10000)
sum_sum = None
for i_episode in range(2000):
    observation = env.reset()
    # print(observation)
    global_error = None
    fake_old_obs = None
    # fake_obs = None
    for t in range(100):
        env.render()
        action = env.action_space.sample()

        old_observation = observation + 0
        if fake_old_obs is None:
            fake_old_obs = old_observation + 0
        observation, reward, done, info = env.step(action)
        # if fake_obs is None:
        #     fake_obs = observation + 0
        last = None
        if len(memory) >= 500:
            # m = search(old_observation, action, observation, memory)
            m = search(fake_old_obs, action, observation, memory)
            fake_old_obs = m.mem.obs + 0
            last = m
            # print(f'Sum diff: {m.sum_diff}\t\tObs diff: {m.obs_diff}')
            if global_error is None:
                global_error = 0
            global_error += m.obs_diff

            memory.append(Mem(old_observation + 0, action + 0, observation + 0))
        else:
            memory.append(Mem(old_observation + 0, action + 0, observation + 0))
        # print(f'len: {len(memory)}')
        # memory.append((old_observation + 0, action + 0, observation + 0))

        # print(reward)
        # print(observation)
        # if done:
        #     # print("Episode finished after {} timesteps".format(t+1))
        #     break
    print(f'Glob error: {global_error}')
    if last is not None:
        print(last.obs_diff)
        print(last.mem.obs)
        print(observation)
env.close()