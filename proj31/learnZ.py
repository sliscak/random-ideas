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
    for result in results[:5]:
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


env = gym.make("procgen:procgen-coinrun-v0")
while True:
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)

    frame = cv2.resize(observation, (128,128))

    patches, bboxes, entropies = gen_patches4(frame)
    min, max = np.min(entropies), np.max(entropies)
    final_image3 = frame * 1
    for bbox, entropy in zip(bboxes, entropies):
        (y, h, x, w) = [int(v) for v in bbox]
        cv2.rectangle(final_image3, (x, y), (w, h), (0, 255, 255), 2)
        # break
        # cv2.rectangle(final_image3, )
    row0_ph[0].image(final_image3, 'marked image', width=130)
    # image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    image = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    hue, saturation, value = image[:,:,0], image[:,:,1], image[:,:,2]
    row3_ph[0].image(hue, caption='hue', width=130)
    row3_ph[1].image(saturation, caption='saturation', width=130)
    row3_ph[2].image(value, caption='value', width=130)
    new_image2 = (hue + saturation + value) // 3
    row3_ph[3].image(new_image2, caption='...', width=130)

    image_copy = image * 1
    image = np.array(image, dtype=int)
    image[:, :, 0] += HUE
    image[:, :, 1] += SATURATION
    # image[:, :, 2] += VALUE
    image = np.clip(image, 0, 255)
    image = image.astype('uint8')

    row1_ph[0].image(image, width=230)
    # image_array_ph.write(image)
    image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    row1_ph[1].image(frame, caption='orig', width=230)

    hue = image[:,:,0]
    hue = cv2.resize(hue, (4, 4))
    hue = cv2.resize(hue, (128, 128))

    saturation = image[:, :, 1]
    saturation = cv2.resize(saturation, (4, 4))
    saturation = cv2.resize(saturation, (128, 128))

    value = image[:, :, 2]
    value = cv2.resize(value, (4, 4))
    value = cv2.resize(value, (128, 128))

    new_image = image * 1
    new_image[:, :, 0] = hue
    new_image[:, :, 1] = saturation
    # new_image[:, :, 2] = value
    # if everyhthing is value the its gray-yellow!
    # new_image = image * 1
    # new_image[:, :, 0] = value
    # new_image = image * 1
    # new_image[:,:,0] = value


    row1_ph[2].image(new_image, caption='low res value applied on image', width=230)

    row2_ph[0].image(image_copy[:, :, 2], caption='lightness', width=230)

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    row2_ph[1].image(gray, caption='gray', width=230)

    # u,s,v = randomized_svd(gray, 20)
    min, max = np.min(gray), np.max(gray)
    u, s, v = randomized_svd(gray, 100)
    # b = u * s
    # b = np.dot(u,v)
    b = np.dot(u * s, v)
    b = (b + np.abs(np.min(b)))
    # b = (b + )
    b /= np.max(b)
    # st.write(b.shape)
    # b = np.reshape(b, (10,10))
    # st.write(b.shape)
    # b = np.reshape(b, (64, -1))
    # b = np.reshape(b, (-1, 128))
    # b = np.reshape(b, (128, -1))
    # b = np.reshape(b, (40, -1))
    # b = np.reshape(b, (128, -1))

    # st.write(u.shape)
    row2_ph[2].image(b, caption='B', width=230)
    # image_array_ph.write(b)


    sleep(1/33)