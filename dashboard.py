import streamlit as st
import pandas as pd
from PIL import Image
import json
import os
import os.path
from streamlit_option_menu import option_menu
import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
import matplotlib.pyplot as plt

def upload_image():
    uploaded_file = st.file_uploader("Upload your Image here", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        sample_image_path = './content/' + uploaded_file.name
        return sample_image_path

# Load Generator model
def load_generator_model(model_json_path, model_weights_path):
    with open(model_json_path, 'r') as json_file:
        loaded_model_json = json_file.read()
    generator_model = model_from_json(loaded_model_json)
    generator_model.load_weights(model_weights_path)
    return generator_model

# Function to preprocess input image
def preprocess_image(image_path):
    img = cv2.imread(image_path).astype("float32")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))  # Resize the image to the expected size
    img = img / 255.0
    return img

# Function to generate colorized image
def generate_colorized_image(generator_model, input_image_path):
    input_image = preprocess_image(input_image_path)
    input_image = np.expand_dims(input_image, axis=0)
    colorized_image = generator_model.predict(input_image)
    colorized_image = np.squeeze(colorized_image, axis=0)
    colorized_image = np.clip(colorized_image, 0, 1)

    # Get the original input image size
    original_image = Image.open(input_image_path)
    original_width, original_height = original_image.size

    # Resize the colorized image to match the original size
    colorized_image = cv2.resize(colorized_image, (original_width, original_height))

    return colorized_image

def Dashboard():
    st.title("Upload Image")
    uploaded_image = upload_image()
    sample_image_path = uploaded_image
    if uploaded_image is not None:
        generator_model = load_generator_model('modelGenerator.json', 'modelGen.h5')
        colorized_image = generate_colorized_image(generator_model, sample_image_path)
        st.image(colorized_image, caption="Colorized Image", use_column_width=True)