import streamlit as st
import tensorflow as tf
from keras.models import load_model
from PIL import Image
import numpy as np

LABELS = {
    0: "apple_fruit",
    1: "banana_fruit",
    2: "cherry_fruit",
    3: "chickoo_fruit",
    4: "grapes_fruit",
    5: "kiwi_fruit",
    6: "mango_fruit",
    7: "orange_fruit",
    8: "strawberry_fruit",
}
IMAGE_SIZE = (128, 128)


def load_and_preprocess_image(image):
    image = image.resize(IMAGE_SIZE)
    image = np.array(image)
    image = image / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)
    return image


def predict(image, model):
    processed_image = load_and_preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_label = np.argmax(prediction)
    return LABELS[predicted_label]


model = load_model("fruit_model.h5")

st.title("Fruit Classification")
st.write("Upload an image of a fruit to classify it.")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    label = predict(image, model)
    st.write(f"The fruit is: {label}")
