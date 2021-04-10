""""
    Human-Assisted/Guided Patch finding.
"""

import streamlit as st
import numpy as np
import cv2
import gym
from time import sleep
# from sklearn.decomposition import TruncatedSVD
from sklearn.utils.extmath import randomized_svd

def gen_patches4(image):
    image = image.astype('float32')
    patches = []
    bboxes=[]
    entropies = []
    for y in range(0,image.shape[0]-KERNEL_SIZE[0], STRIDE):
        for x in range(0,image.shape[1]-KERNEL_SIZE[1], STRIDE):
            patch = image[y:y+KERNEL_SIZE[0], x:x+KERNEL_SIZE[1]]
            entropy = np.mean(np.abs(np.mean(patch) - patch))
            # patches.append(patch)
            bbox = (y, y + KERNEL_SIZE[0], x, x + KERNEL_SIZE[1])
            patches.append(patch)
            bboxes.append(bbox)
            entropies.append(entropy)

    return patches, bboxes, entropies


def gen_patches5(image): # use topk=10?
    image = image.astype('float32')
    patches = []
    bboxes=[]
    entropies = []
    results = []
    for y in range(0,image.shape[0]-KERNEL_SIZE[0], STRIDE):
        for x in range(0,image.shape[1]-KERNEL_SIZE[1], STRIDE):
            patch = image[y:y+KERNEL_SIZE[0], x:x+KERNEL_SIZE[1]]
            entropy = np.mean(np.abs(np.mean(patch) - patch))
            # patches.append(patch)
            bbox = (y, y + KERNEL_SIZE[0], x, x + KERNEL_SIZE[1])
            # patches.append(patch)
            # bboxes.append(bbox)
            # entropies.append(entropy)
            results.append((patch, bbox, entropy))
            # results.append([patch, bbox, entropy])
    results.sort(key=lambda x: x[2],reverse=True)
    # for result in results[:10]:
    for result in results:
        patch, bbox, entropy = result
        patches.append(patch)
        bboxes.append(bbox)
        entropies.append(entropy)
    return patches, bboxes, entropies

def gen_patches6(image): # use topk=10?
    image = image.astype('float32')
    patches = []
    bboxes=[]
    entropies = []
    results = []
    for y in range(0,image.shape[0]-KERNEL_SIZE[0], STRIDE):
        for x in range(0,image.shape[1]-KERNEL_SIZE[1], STRIDE):
            patch = image[y:y+KERNEL_SIZE[0], x:x+KERNEL_SIZE[1]]
            # entropy = np.mean(np.abs(np.mean(patch) - patch))
            entropy = np.abs(np.mean(patch) - patch) # per pixel entropies
            entropy = np.mean(np.abs(np.mean(entropy) - entropy)) # whole patch entropy
            # patches.append(patch)
            bbox = (y, y + KERNEL_SIZE[0], x, x + KERNEL_SIZE[1])
            # patches.append(patch)
            # bboxes.append(bbox)
            # entropies.append(entropy)
            results.append((patch, bbox, entropy))
            # results.append([patch, bbox, entropy])
    results.sort(key=lambda x: x[2],reverse=True)
    # for result in results[:10]:
    for result in results:
        patch, bbox, entropy = result
        patches.append(patch)
        bboxes.append(bbox)
        entropies.append(entropy)
    return patches, bboxes, entropies

def gen_patches7(image): # use topk=10?
    image = image.astype('float32')
    patches = []
    bboxes=[]
    entropies = []
    results = []
    for y in range(0,image.shape[0]-KERNEL_SIZE[0], STRIDE):
        for x in range(0,image.shape[1]-KERNEL_SIZE[1], STRIDE):
            patch = image[y:y+KERNEL_SIZE[0], x:x+KERNEL_SIZE[1]]
            # entropy = np.mean(np.abs(np.mean(patch) - patch))
            entropy = np.abs(np.median(patch) - patch) # per pixel entropies
            entropy = np.median(np.abs(np.median(entropy) - entropy)) # whole patch entropy
            # patches.append(patch)
            bbox = (y, y + KERNEL_SIZE[0], x, x + KERNEL_SIZE[1])
            # patches.append(patch)
            # bboxes.append(bbox)
            # entropies.append(entropy)
            results.append((patch, bbox, entropy))
            # results.append([patch, bbox, entropy])
    results.sort(key=lambda x: x[2],reverse=True)
    # for result in results[:10]:
    for result in results:
        patch, bbox, entropy = result
        patches.append(patch)
        bboxes.append(bbox)
        entropies.append(entropy)
    return patches, bboxes, entropies

def gen_patches8(image): # use topk=10?
    image = image.astype('float32')
    patches = []
    bboxes=[]
    entropies = []
    results = []
    for y in range(0,image.shape[0]-KERNEL_SIZE[0], STRIDE):
        for x in range(0,image.shape[1]-KERNEL_SIZE[1], STRIDE):
            patch = image[y:y+KERNEL_SIZE[0], x:x+KERNEL_SIZE[1]]
            # entropy = np.mean(np.abs(np.mean(patch) - patch))
            entropy = np.abs(np.mean(patch) - patch) # per pixel entropies
            entropy = np.max(entropy)
            # entropy = np.median(np.abs(np.median(entropy) - entropy)) # whole patch entropy
            # patches.append(patch)
            bbox = (y, y + KERNEL_SIZE[0], x, x + KERNEL_SIZE[1])
            # patches.append(patch)
            # bboxes.append(bbox)
            # entropies.append(entropy)
            results.append((patch, bbox, entropy))
            # results.append([patch, bbox, entropy])
    results.sort(key=lambda x: x[2],reverse=True)
    # for result in results[:10]:
    for result in results:
        patch, bbox, entropy = result
        patches.append(patch)
        bboxes.append(bbox)
        entropies.append(entropy)
    return patches, bboxes, entropies

threshold_values = [x/10 for x in range(0, 12)]
threshold_values.insert(0, None)
threshold_values.extend([1 + (0.5 * x) for x in range(1, 10)])

THRESHOLD = st.sidebar.selectbox(
    'Choose threshold value', options=threshold_values, index=6)

kernel_sizes = [(x,x) for x in range(1, 33)]
KERNEL_SIZE = st.sidebar.selectbox(
    'Choose kernel size', options=kernel_sizes, index=4)

stride_vals = [x for x in range(1, 21)]
STRIDE = st.sidebar.selectbox(
    'Choose stride value', options=stride_vals, index=0)

padding_vals = [x for x in range(11)]
PADDING = st.sidebar.selectbox(
    'Choose padding value', options=padding_vals, index=10)

row0_ph = [col.empty() for col in st.beta_columns(3)]
row1_ph = [col.empty() for col in st.beta_columns(3)]
row2_ph = [col.empty() for col in st.beta_columns(3)]
row3_ph = [col.empty() for col in st.beta_columns(4)]
image_array_ph = st.empty()

# cap = cv2.VideoCapture(0)

# ret, old_frame = cap.read()
#
hue_vals = range(-250, 251)
# hue_vals = np.array(hue_vals, dtype='uint8')
HUE = st.sidebar.selectbox(
    'Choose HUE value', options=hue_vals, index=250)

saturation_vals = range(-250, 251)
# saturation_vals = np.array(saturation_vals, dtype='uint8')
SATURATION = st.sidebar.selectbox(
    'Choose SATURATION value', options=saturation_vals, index=250)

value_vals = range(-250, 251)
# value_vals = np.array(value_vals, dtype='uint8')
VALUE = st.sidebar.selectbox(
    'Choose VALUE value', options=value_vals, index=250)

def match_image(image, patch):
    KERNEL_SIZE = patch.shape
    STRIDE = 1
    found_best = None
    for y in range(0,image.shape[0]-KERNEL_SIZE[0], STRIDE):
        for x in range(0,image.shape[1]-KERNEL_SIZE[1], STRIDE):
            patch2 = image[y:y + KERNEL_SIZE[0], x:x + KERNEL_SIZE[1]]
            diff = np.sum(np.abs(patch, patch2))
            # if len(found_best) == 0:
            #     found_best = (diff, patch2)
            if found_best is None:
                found_best = (diff, patch2)
            else:
                if diff <= found_best[0]:
                    found_best = (diff, patch2)
    print('matched!')
    return found_best

env = gym.make("procgen:procgen-coinrun-v0")
old_frame = None
cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
cv2.namedWindow('Patch', cv2.WINDOW_NORMAL)
cv2.namedWindow('Found Patch', cv2.WINDOW_NORMAL)
# cv2.namedWindow('Changed', cv2.WINDOW_NORMAL)
bbox = None
patch = None
while True:
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    frame = cv2.resize(observation, (128, 128))
    image1 = frame * 1
    image2 = frame * 1
    changed = frame * 1
    # image = frame[::2,::2,:]
    image1[:,::2,:] = 0
    image2[::2,:,:] = 0
    cv2.imshow("Frame", frame)
    if patch is not None:
        cv2.imshow('Patch', patch)
        diff, found_patch = match_image(frame, patch)
        cv2.imshow('Found Patch', found_patch)
    # cv2.imshow("Image1", image1)
    # cv2.imshow("Image2", image2)
    # if bbox is not None:
    #     p1 = (int(bbox[0]), int(bbox[1]))
    #     p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    #     # changed[p1[1]:p2[1], p1[0]:p2[0],:] = 0
    #     patch = frame[p1[1]:p2[1], p1[0]:p2[0],:]

    cv2.imshow('Changed', changed)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        bbox = cv2.selectROI("Frame", frame, fromCenter=False,
                               showCrosshair=True)
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        patch = frame[p1[1]:p2[1], p1[0]:p2[0], :]
        # print(bbox[:2], bbox[3::])
        # breakpoint()
        # cv2.rectangle(changed, (0, 10), (50, 70), (0, 255, 0), 2)
        # st.write(bbox)
    sleep(1/33)