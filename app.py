import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import pathlib
from tensorflow import keras

st.title("Emotion Detection")

st.write("Predict the emotion being represented in the live image.")

model = keras.models.load_model("model.h5")

labels = [
    "angry",
    "disgusted",
    "fearful",
    "happy",
    "neutral",
    "sad",
    "surprised"
]

# Function to predict emotion from an image
def predict_emotion(image):
    image = tf.image.resize(image, (112, 112))
    img_array = tf.image.convert_image_dtype(image, tf.float32)
    img_array = tf.image.rgb_to_grayscale(img_array)
    img_array = tf.image.per_image_standardization(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    label = labels[np.argmax(predictions)]
    return label

st.write("### Prediction Result")

if st.button("Predict"):
    st.write("Please allow access to your camera.")
    video_stream = st.camera()

    if video_stream:
        st.image(video_stream, caption="Live Image", use_column_width=True, channels="BGR")
        image_array = np.array(video_stream)
        label = predict_emotion(image_array)
        st.markdown(
            f"<h2 style='text-align: center;'>Predicted Emotion: {label}</h2>",
            unsafe_allow_html=True,
        )

# If you would not like to use the webcam, you can use a sample image instead.
sample_img_choice = st.button("Use Sample Image")

if sample_img_choice:
    image1 = Image.open("seed_charlock.png")
    image1 = image1.resize((256, 256))
    image_array = np.array(image1)
    label = predict_emotion(image_array)
    st.markdown(
        f"<h2 style='text-align: center;'>Predicted Emotion: {label}</h2>",
        unsafe_allow_html=True,
    )
