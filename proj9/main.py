import torch
import numpy as np
import cv2

from copy import deepcopy
from time import sleep

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.namedWindow('image2', cv2.WINDOW_NORMAL)

delay = 1

while True:
    image = torch.rand((1280, 720)).numpy()
    cv2.imshow('image', image)
    cv2.waitKey(1)