import streamlit as st
import os
from tensorflow import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.preprocessing import image
from PIL import Image
import warnings
warnings.filterwarnings('ignore')
import pathlib
from keras.models import load_model
import matplotlib.image as mpimg

st.title("emotion detection")

st.write("Predict the emotion that is being represented in the image.")

model = load_model("model.h5")
l=[ 'angry',
    'disgusted',
    'fearful',
    'happy',
    'neutral',
    'sad',
    'surprised']
uploaded_file = st.file_uploader(
    "Upload an image of a emotion :", type="jpg"
)
predictions=-1
if uploaded_file is not None:
    image1 = Image.open(uploaded_file)
    image1=image.smart_resize(image1,(112,112))
    img_array = image.img_to_array(image1)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array/255.0
    predictions = model.predict(img_array)
    label=l[np.argmax(predictions)]


st.write("### Prediction Result")
if st.button("Predict"):
    if uploaded_file is not None:
        image1 = Image.open(uploaded_file)
        st.image(image1, caption="Uploaded Image", use_column_width=True)
        st.markdown(
            f"<h2 style='text-align: center;'>Image of {label}</h2>",
            unsafe_allow_html=True,
        )
    else:
        st.write("Please upload file or choose sample image.")


st.write("If you would not like to upload an image, you can use the sample image instead:")
sample_img_choice = st.button("Use Sample Image")

if sample_img_choice:
    image1 = Image.open("seed_charlock.png")
    image1=image.smart_resize(image1,(256,256))
    img_array = image.img_to_array(image1)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array/255.0
    predictions = model.predict(img_array)
    label=l[np.argmax(predictions)]
    st.markdown(
        f"<h2 style='text-align: center;'>{label}</h2>",
        unsafe_allow_html=True,
    )
