import cv2
import gym
import numpy as np
import torch
from proj1.model import Net
from time import sleep
from collections import deque
cv2.namedWindow('main', cv2.WINDOW_NORMAL)
cv2.namedWindow('diff', cv2.WINDOW_NORMAL)
cv2.namedWindow('pozadie', cv2.WINDOW_NORMAL)
cv2.namedWindow('diff2', cv2.WINDOW_NORMAL)

# class Plan:
#     def __init__(self):
#
#     def show(self):

def main():
    net = Net()
    criterion = torch.nn.MSELoss(reduction='sum')
    # optimizer = torch.optim.SGD(net.parameters(), lr=1e-4, momentum=0.9)
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.0000001)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
    env = gym.make("procgen:procgen-climber-v0", start_level=0, num_levels=1)
    print(env.action_space)
    # net = None
    for i_episode in range(20):
        observation = env.reset()
        plan = [env.action_space.sample() for i in range(5)]
        prev_gs_image = None
        pozadie_rolling_average = None

        for t in range(10000):
            env.render()
            #print(observation)
            # bgr_image = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)
            gs_image = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY) / 255
            # gs_image = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR) / 255
            if prev_gs_image is None:
                prev_gs_image = gs_image * 0
            # if net is None:
            #     net = Net(gs_image.shape)
            # pred_img = net(gs_image)
            # print(pred_img.shape)
            # loss = criterion(pred_img, torch.Tensor(gs_image))
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            # cv2.imshow('main', np.array(pred_img.detach()))
            gs = np.array(gs_image)

            pgs = np.array(prev_gs_image)
            # diff = (((pgs - gs)**2) > 0) * gs
            # diff = pgs - gs
            diff = ~((pgs - gs) == 0) * gs
            pozadie = ((pgs - gs) == 0) * gs
            if pozadie_rolling_average is None:
                pozadie_rolling_average = pozadie
            pozadie_rolling_average += pozadie
            # pozadie_rolling_average = (pozadie_rolling_average + pozadie) / 2
            pra = pozadie_rolling_average / t
            # diff2 = ((pra - gs) > 0) * gs
            diff2 = (pra - gs) **2
            cv2.imshow('main', gs)
            cv2.imshow('diff', diff)
            cv2.imshow('diff2', diff2)
            cv2.imshow('pozadie', pra)
            cv2.waitKey(100)
            prev_gs_image = gs_image
            # action = env.action_space.sample()
            if len(plan) > 0:
                print(f'PLAN: {plan}')
                action = plan[0]
                del plan[0]
            else:
                plan = [env.action_space.sample() for i in range(5)]
                action = plan[0]
                del plan[0]
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
    env.close()


if __name__ == "__main__":
    main()