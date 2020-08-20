# TESTING PATTERN MATCHER

import torch
import numpy as np
import cv2
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.feature_extraction.image import extract_patches_2d
from random import sample
from time import sleep


def plot_patches(patches):
    pass


image = Image.open('image.png')
image = np.array(image)  # [::4, ::4]


def downsample(image, shape):
    return image[::shape[0], ::shape[1]]


image = downsample(image, (6, 6))
print(image.shape)
# exit()
plt.imshow(image)
plt.show()
# patches = extract_patches_2d(image, (100, 100), max_patches=0.05)
# print(patches.shape)
# print(patches[0])
# plt.imshow(patches[89000])
# plt.show()
# plt.plot(patches)
patch = image[0:100, 0:100]
print(patch)
plt.imshow(patch)
plt.show()


# STRIDE?? PADDING??
def make_patch_generator(image, shape, stride=1):
    x_max, y_max, c_max = image.shape
    for x in range(0, x_max - shape[0], stride):
        for y in range(0, y_max - shape[1], stride):
            yield image[x:x + shape[0], y:y + shape[1]]
    # for x in range(0, x_max-shape[0], shape[0]):
    #     for y in range(0, y_max-shape[1], shape[1]):
    #         yield image[x:x+shape[0], y:y+shape[1]]


# def patch_generator(image, shape):
#     x_max, y_max, c_max = image.shape
#     for y in range(y_max-shape[1]):
#         for x in range(x_max-shape[0]):
#             yield image[x:shape[0], y:shape[1]]

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.namedWindow('inverse_image', cv2.WINDOW_NORMAL)
cv2.namedWindow('match_patch', cv2.WINDOW_NORMAL)
cv2.namedWindow('matching', cv2.WINDOW_NORMAL)

KERNEL_SIZE = (50, 50)
m_patch = cv2.cvtColor(image[50:50 + KERNEL_SIZE[0], 50:50 + KERNEL_SIZE[1]], cv2.COLOR_RGB2BGR)
patch_generator = make_patch_generator(image, KERNEL_SIZE, 2)
matchlist = []
for patch in patch_generator:
    patch = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
    cv2.imshow('image', patch)
    cv2.imshow('inverse_image', 255 - patch)
    cv2.imshow('match_patch', m_patch)
    # cv2.imshow('matching', 1 - np.abs((255-m_patch) - (255-patch)))
    cv2.imshow('matching', cv2.applyColorMap(255 - np.abs((255 - m_patch) - (255 - patch)), cv2.COLORMAP_JET))
    # cv2.imshow('matching', cv2.applyColorMap((1-((np.abs(m_patch - patch)/ 255)) * 255), cv2.COLORMAP_JET))
    # matchlist.append(np.mean(np.abs((255 - m_patch) - (255 - patch)) / 255))
    match_val = np.mean(1 - (np.abs(m_patch - patch) / 255))
    if match_val > 0.98:
        print(f'Found match: {match_val} similarity')
        cv2.waitKey(6000)
    matchlist.append(match_val)
    cv2.waitKey(1)
    # sleep(1)
matchlist.sort(reverse=True)
print(matchlist[0:10])

# patches = [patch for patch in patch_generator]
# print(patches)

# 5, 0, 255
# 0 == 1, 255 == 0,
# 0 + 1, =
# 0 == 1, 1 == 0

#
# 200, 150 = 200 - 150 = 50
#          = 150 - 200 = -50
#          = abs
#          =
