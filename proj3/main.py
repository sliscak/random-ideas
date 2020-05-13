import torch
import cv2




from random import randint, shuffle, random, SystemRandom
from copy import deepcopy

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.namedWindow('image2', cv2.WINDOW_NORMAL)
SR = SystemRandom()

def dunkle2():
    image = torch.ones(size=(64 * 64 * 1,), dtype=float)
    while True:
        # position = randint(0, (64 * 64) - 1)
        position = SR.randint(0, (64 * 64) - 1)
        r_image = deepcopy(image)
        r_image[position] = 0
        dist = torch.cosine_similarity(image, r_image, dim=0)
        while SR.random() < dist:
            position = SR.randint(0, (64 * 64) - 1)
            r_image[position] = 0
            dist = torch.cosine_similarity(image, r_image, dim=0)
        cv2.imshow('image', r_image.reshape((64, 64, 1)).numpy())
        cv2.waitKey(100)

# def dunkle():
#     image = torch.ones(size=(64 * 64 * 1,), dtype=float)
#     while True:
#         # position = randint(0, (64 * 64) - 1)
#         position = SR.randint(0, (64 * 64) - 1)
#         r_image = deepcopy(image)
#         r_image[position] = 0
#         dist = torch.cosine_similarity(image, r_image, dim=0)
#         while SR.random() < dist:
#             position = SR.randint(0, (64 * 64) - 1)
#             r_image[position] = 0
#             dist = torch.cosine_similarity(image, r_image, dim=0)
#         cv2.imshow('image', r_image.reshape((64, 64, 1)).numpy())
#         cv2.waitKey(100)

dunkle2()
#
# # imgs = [torch.randint(low=0, high=2, size=(32, 32, 1), dtype=float) for i in range(1000)]
# # i = 0
# # image = torch.randint(low=0, high=2, size=(32, 32, 1), dtype=float)
# image = torch.ones(size=(32*32*3,), dtype=float)
# orig = list(range(32*32*3))
# image2 = torch.randint(low=0, high=2, size=(32*32,), dtype=float)
# n = 0


# while True:
#     cv2.imshow('image',image.reshape((32, 32, 3)).numpy())
#     cv2.imshow('image2', image2.reshape((32, 32)).numpy())
#     cv2.waitKey(100)
#     image = torch.ones(size=(32 * 32 * 3,), dtype=float)
#     image2 = torch.randint(low=0, high=2, size=(32 * 32,), dtype=float)
#     n = randint(0, (32 * 32 * 3) - 1)
#     non = deepcopy(orig)
#     shuffle(non)
#     non = non[0:n]
#     # n = randint(0, (32*32) - 1)
#     print(non)
#     print(f'N: {n}')
#     for i in non:
#         image[i] = random()
    # rn = set(
    #     [randint(0, (32*32) -1) for x in range(n)]
    # )
    # n += 1
    # if n >= 32 * 32:
    #     n = 0
    # rc = 0
    # for i in rn:
    #     rc += 1
    #     image[i] = 0
    # print(f'RC: {rc}')
    # rc2 = image2.sum()
    # print(f'RC2: {rc2}')
    # image[r] = randint(0, 1)
    # r = randint(0, 32*32)
    # image = torch.ones(size=(32, 32, 1), dtype=float)
    # for i in range(r):
    #     position = (randint(0, 127), randint(0, 127))
    #     image[position] = 0
        # image *= torch.randint(low=0, high=2, size=(32, 32, 1), dtype=float)
    # position = torch.randint(0, 32, size=(2,))
    # position = (randint(0, 127), randint(0, 127))
    # image[position] = randint(0, 1)