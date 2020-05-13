import cv2
import gym
import numpy as np
import torch
from proj1.model10 import Net
from time import sleep
from collections import deque
cv2.namedWindow('main', cv2.WINDOW_NORMAL)
cv2.namedWindow('net', cv2.WINDOW_NORMAL)

# confidence level
# classifier
# classify 32*32*3 flattened image

def PSNR(mse):
    psnr = -10 * torch.log10(1 / mse)
    return psnr

def train(net, optimizer, scaled_image, device, enabled):
    criterion = torch.nn.MSELoss()  # torch.nn.MSELoss(reduction='sum')
    x_train = torch.Tensor(scaled_image).flatten().to(device)
    y_train = torch.Tensor(scaled_image).to(device)
    loss = net(x_train, False)
    optimizer.zero_grad()
    print(f'Loss: {loss.detach()}')
    if enabled is True:
        loss.backward()
        optimizer.step()
    else:
        print(f'Not learning!')
    cv2.imshow('main', y_train.detach().cpu().numpy())
    cv2.imshow('net', net(x_train, True).detach().cpu().reshape((32,32,3)).numpy())
    # cv2.imshow('net', y_pred.detach().cpu().numpy())
    return loss.detach().cpu()

def main():
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    dev = 'cpu'
    device = torch.device(dev)
    net = Net().to(device)
    # optimizer = torch.optim.SGD(params=net.parameters(), lr=0.01)
    optimizer = torch.optim.Adam(params=net.parameters(), lr=0.0001)

    env = gym.make("procgen:procgen-heist-v0", start_level=0, num_levels=1)
    enabled = True
    for i_episode in range(20):
        env = gym.make("procgen:procgen-heist-v0", start_level=0, num_levels=1)
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