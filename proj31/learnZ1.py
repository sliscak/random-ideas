import streamlit as st
import numpy as np
import cv2


image_ph, image_ph2 = [col.empty() for col in st.beta_columns(2)]
image_array_ph = st.empty()

cap = cv2.VideoCapture(0)

ret, old_frame = cap.read()

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

while True:
    ret, frame = cap.read()
    if ret is True:
        frame = cv2.resize(frame, (128,128))
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        image = np.array(image, dtype=int)
        image[:, :, 0] += HUE
        image[:, :, 1] += SATURATION
        image[:, :, 2] += VALUE
        image = np.clip(image, 0, 255)
        image = image.astype('uint8')

        image_ph.image(image, width=230)
        image_array_ph.write(image)
        image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        image_ph2.image(image, width=230)