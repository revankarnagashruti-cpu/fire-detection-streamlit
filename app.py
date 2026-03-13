import streamlit as st
import numpy as np
import cv2
from PIL import Image

st.title("🔥 Fire Detection Demo")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    if st.button("Detect Fire"):
        st.success("Prediction: Fire Detected (Demo)")
