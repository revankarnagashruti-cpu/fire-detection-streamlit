import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

# Create demo model
model = Sequential([
    Flatten(input_shape=(64,64,3)),
    Dense(128, activation='relu'),
    Dense(2, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

st.title("🔥 Fire Detection System")

st.write("Upload an image to check fire detection")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    img = np.array(image)
    img = cv2.resize(img,(64,64))
    img = np.expand_dims(img, axis=0)

    if st.button("Detect Fire"):

        prediction = model.predict(img)

        class_id = np.argmax(prediction)

        if class_id == 0:
            st.success("Normal Image")
        else:
            st.error("Fire Detected")
