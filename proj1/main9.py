import cv2
import gym
import numpy as np
import torch
from proj1.model5 import Net
from time import sleep
from collections import deque
cv2.namedWindow('main', cv2.WINDOW_NORMAL)
cv2.namedWindow('diff', cv2.WINDOW_NORMAL)
cv2.namedWindow('pozadie', cv2.WINDOW_NORMAL)
cv2.namedWindow('diff2', cv2.WINDOW_NORMAL)
cv2.namedWindow('net', cv2.WINDOW_NORMAL)

# class Plan:
#     def __init__(self):
#
#     def show(self):

def main():
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    dev = 'cpu'
    device = torch.device(dev)

    net = Net().to(device)
    criterion = torch.nn.MSELoss()#torch.nn.MSELoss(reduction='sum')
    # optimizer = torch.optim.Adam(net.parameters(), lr=0.00001)
    optimizer = torch.optim.SGD(params=net.parameters(), lr=0.01)
    env = gym.make("procgen:procgen-heist-v0", start_level=0, num_levels=1)
    print(env.action_space)
    # net = None
    train = True
    for i_episode in range(20):
        env = gym.make("procgen:procgen-heist-v0", start_level=0, num_levels=1)
        observation = env.reset()
        last_image = None
        for t in range(10000):
            env.render()
            gs_image = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)
            # resize and scale image to values between 0 and 1
            gs_image = cv2.resize(gs_image, dsize=(32,32)) / 255
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            for u in range(1):
                if last_image is None:
                    last_image = np.array(gs_image)
                loss_sum = 0
                # diff = ~((last_image - gs_image) == 0) * gs_image
                diff = ((last_image - gs_image) != 0) * gs_image
                masked_img = gs_image * (diff == 0)
                pred_volume = net(torch.Tensor(gs_image).flatten().to(device))
                # y_volume = torch.Tensor(gs_image).unsqueeze(0).repeat_interleave(4, dim=0).to(device)
                y_volume = torch.Tensor(masked_img).to(device)
                loss = criterion(pred_volume, y_volume)
                psnr = -10 * torch.log10(1 / loss)
                loss_sum += psnr.detach()
                optimizer.zero_grad()
                if train and (loss.detach() > 0.000000005):
                    psnr.backward()
                    optimizer.step()
                    print(f'grad: {net.p.grad}')
                else:
                    print('Not training!')
                    train = False
                cv2.waitKey(1)
                print(u)
                # average loss where?
                # average loss where?
                print(f'Loss: {loss.detach()}')
                # print(f'PSNR sum: {psnr_sum}')
                # pred_image = np.zeros((32, 32, 3))
                pred_volume = net(torch.Tensor(gs_image).flatten().to(device)).detach().cpu().numpy()
                cv2.imshow('net', pred_volume)
                gs = np.array(gs_image)
                cv2.imshow('main', gs)
                cv2.imshow('diff', diff)
                last_image = np.array(gs_image)
                # cv2.imshow('net', np.array(pred_img.detach()))
                cv2.waitKey(100)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
    env.close()


if __name__ == "__main__":
    main()