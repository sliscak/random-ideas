import torch
import numpy as np
import cv2

from itertools import permutations, product
from random import randint, shuffle, random, SystemRandom
from copy import deepcopy

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.namedWindow('image2', cv2.WINDOW_NORMAL)

# array = [chr(n) for n in range(48, 60)]
#
# for p in permutations(array, 3):
#     print(p)

array0 = [0 for x in range(32)]
array1 = [1 for x in range(32)]

# array = array0 + array1
array = [0, 1]
i = 0
for p in product(array, repeat=32*32):
    if (i % 10000) == 0:
        image = np.array(p, dtype=float).reshape(32, 32)
        cv2.imshow('image', image)
        cv2.waitKey(1)
    i += 1
    print(f'Images: {i}')
# 32 * 32 = 1 black image
# 1 changed pixel on 32 * 32 = 32 * 32 possibilities
# 2 changed pixels on 32 * 32 = (32 * 32) + (32 * 32)