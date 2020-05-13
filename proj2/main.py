import cv2
import gym
import numpy as np
import torch
from proj2.model import Cleaner, Fusher
from time import sleep
from collections import deque
cv2.namedWindow('main', cv2.WINDOW_NORMAL)
cv2.namedWindow('net', cv2.WINDOW_NORMAL)

# confidence level
# classifier
# classify 32*32*3 flattened image

# source Pytorch DOCS
def PSNR(mse):
    psnr = -10 * torch.log10(1 / mse)
    return psnr

def train(net, optimizer, scaled_image, device, enabled):
    pass

def main():
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    dev = 'cpu'
    device = torch.device(dev)
    net = Net().to(device)
    # optimizer = torch.optim.SGD(params=net.parameters(), lr=0.01)
    optimizer = torch.optim.Adam(params=net.parameters(), lr=0.000001)

    # env = gym.make("procgen:procgen-heist-v0", start_level=0, num_levels=1)
    enabled = True
    for i_episode in range(2000):
        env = gym.make("procgen:procgen-starpilot-v0", start_level=0, num_levels=1)
        observation = env.reset()
        for t in range(10000):
            env.render()
            bgr_image = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)
            # resize and scale image to values between 0 and 1
            scaled_image = cv2.resize(bgr_image, dsize=(32,32)) / 255
            loss = train(net, optimizer, scaled_image, device, enabled)
            # if loss < 0.0007:
            #     enabled = False
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            cv2.waitKey(1)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
    env.close()


if __name__ == "__main__":
    main()