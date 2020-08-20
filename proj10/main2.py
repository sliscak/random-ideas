# using reservoirs for image generation, superresolution.

import torch
import numpy as np
import cv2
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.feature_extraction.image import extract_patches_2d
from proj10.model import RandomNet, Net
from random import sample
from time import sleep


image = Image.open('image.png')
image = image.resize((32, 32), resample=Image.NEAREST)
# image = np.array(image)#[::4, ::4]
# def downsample(image, shape):
#     return image[::shape[0], ::shape[1]]
# # down_image = downsample(image, (16, 16))
image = torch.Tensor(np.array(image))/255
print(image.shape)
print(image)
# down_image = torch.nn.functional.interpolate(image, size=(32, 32))

rnet = RandomNet()
net = Net()
optimizer = torch.optim.Adam(params=rnet.parameters(), lr=0.0001)
criterion = torch.nn.MSELoss()

tensor = torch.ones((32)).detach()
# tensor = torch.randn((32,)).detach()

cv2.namedWindow('orig_image', cv2.WINDOW_NORMAL)
cv2.namedWindow('neural_image', cv2.WINDOW_NORMAL)
while True:
    latent = net(tensor)
    net_img = rnet(latent)
    loss = criterion(net_img, image.detach())
    print(f'loss: {loss}')
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    cv2.imshow('orig_image', cv2.cvtColor(image.numpy(), cv2.COLOR_RGB2BGR))
    cv2.imshow('neural_image', cv2.cvtColor(net_img.detach().numpy(), cv2.COLOR_RGB2BGR))
    print(f'Latent: {latent.detach()}')
    cv2.waitKey(100)

