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

def search(old_obs, action, memory):
    scores = []
    for m in list(memory)[0:2000]:
        # oo, a, o = m.old_obs, m.action, m.obs
        score1 = np.sum((m.old_obs - old_observation) ** 2)
        score2 = np.sum((m.action - action) ** 2)
        score3 = np.sum((m.obs - observation) ** 2)
        scores.append(Hit(score1, score2, score3, m))
        # scores.append((score1, score2, score3))
        # score1 = np.sum((oo - old_observation) ** 2)
        # score2 = (a - action) ** 2
        # scores.append((score1, score2))
    scores.sort(key=lambda hit: hit.old_obs_diff + hit.action_diff)
    # print(scores[0])
    print(f'Pred: {scores[0]}\nTruth: {(old_observation, action, observation)}')


Hit = namedtuple('Hit', 'old_obs_diff action_diff obs_diff mem')
Mem = namedtuple('Mem', 'old_obs action obs')

env = gym.make('MountainCar-v0')
memory = deque(maxlen=10000)
for i_episode in range(2000):
    observation = env.reset()
    print(observation)
    for t in range(10000):
        env.render()
        action = env.action_space.sample()

        old_observation = observation + 0
        observation, reward, done, info = env.step(action)
        if len(memory) > 4000:
            # hits = []
            scores = []
            for m in list(memory)[0:2000]:
                # oo, a, o = m.old_obs, m.action, m.obs
                score1 = np.sum((m.old_obs - old_observation) ** 2)
                score2 = np.sum((m.action - action) ** 2)
                score3 = np.sum((m.obs - observation) ** 2)
                scores.append(Hit(score1, score2, score3, m))
                # scores.append((score1, score2, score3))
                # score1 = np.sum((oo - old_observation) ** 2)
                # score2 = (a - action) ** 2
                # scores.append((score1, score2))
            scores.sort(key=lambda hit: hit.old_obs_diff + hit.action_diff)
            # print(scores[0])
            print(f'Pred: {scores[0]}\nTruth: {(old_observation, action, observation)}')
            exit()
        print(f'len: {len(memory)}')
        # memory.append((old_observation + 0, action + 0, observation + 0))
        memory.append(Mem(old_observation+0, action+0, observation+0))


        print(reward)
        print(observation)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()