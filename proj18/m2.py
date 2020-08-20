import gym
import numpy as np
import torch
#from torch.nn import modules as nn
from torch import nn
from collections import deque

''''
    INPUT: 10
    MEM_STATE: 13
    OUTPUT: 3
'''
class GameNet(nn.Module):
    def __init__(self):
        super(GameNet, self).__init__()
        self.l1 = nn.Sequential(
            nn.Linear((3*3)+1+13, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 3),
            nn.Tanh(),
        )

    def forward(self, input, mem_state):
        x = torch.cat((input, mem_state), dim=0).float()
        out = self.l1(x)
        return out

gamenet = GameNet()
optimizer = torch.optim.Adam(params=gamenet.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

env = gym.make('Pendulum-v0')
for i_episode in range(20):
    observation = env.reset()
    state_d = deque([np.zeros(3) for x in range(3)], maxlen=3)
    state_d.append(observation)
    memory = deque(maxlen=1000)
    for t in range(100):
        env.render()
        action = env.action_space.sample()
        state = torch.flatten(torch.tensor(state_d))
        input = torch.cat((state, torch.tensor(action)))

        observation, reward, done, info = env.step(action)
        next_state = torch.flatten(torch.Tensor(observation))
        mem_state = torch.cat((input, next_state))

        comp_next_state = gamenet(input, mem_state)
        loss = criterion(comp_next_state, next_state.detach())
        print(f'Input: {input.detach()}\nOutput: {comp_next_state.detach()}\nLoss: {loss.detach()}')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(f'action: {action}')
        # memory.append((input, torch.tensor(observation)))
        # if len(memory) >= 20:
        #     inp, act = memory[-1]
        #     output = gamenet(inp)
        #     loss = criterion(output, act.detach())
        #     print(f'Input: {inp.detach()}\nOutput: {output.detach()}\nLoss: {loss.detach()}')
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()
        # d.append(observation)

        print(reward)
        print(observation)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()