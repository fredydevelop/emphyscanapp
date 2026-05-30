import numpy as np
import pandas as pd
import tensorflow as tf
from keras.utils import to_categorical
import random
import os
import imghdr
import streamlit as st
import pickle as pk
import cv2
import requests
from PIL import Image
from io import BytesIO
import streamlit as st
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.image import resize
from tensorflow.keras.models import load_model, save_model


col1, col2 = st.columns(2)

with col1:
    st.image("emphylogo.png", width=120)
    st.header("Emphysema Chest Xray Detector")
    #st.image("MMJXray.jpg",width=80)




class_labels = {
    0: "Emphysema",
    1: "Normal",
}

def get_recommendation(predicted_category):
    if predicted_category == "Emphysema":
        return (
            "The chest X-ray shows features consistent with emphysema. "
            "This AI prediction is not a medical diagnosis. Please consult a qualified healthcare professional "
            "or radiologist for further evaluation, confirmation, and appropriate management."
        )

    elif predicted_category == "Normal":
        return (
            "The chest X-ray appears normal with no signs suggestive of emphysema detected by the model. "
            "If symptoms such as shortness of breath, chronic cough, or wheezing persist, "
            "seek medical advice for a comprehensive assessment."
        )

    else:
        return "No recommendation available."
    


def insert():
    #st.markdown("Emphysema Chest Xray clasification system")
    
    # File uploader widget
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png","bmp"], key="upl")

    if uploaded_file is not None:
        # Convert the uploaded image to RGB
        img = Image.open(uploaded_file).convert("RGB")
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add a batch dimension
        # To load the model
        loaded_model = load_model("/main_emphysema_model.keras")
        # Make the prediction
        st.subheader("Uploaded Image")
        st.image(img, width=400)
        
        if st.button("Predict"):
            prediction = loaded_model.predict(img_array)
            predicted_class = int(np.argmax(prediction, axis=1)[0])
            confidence = float(np.max(prediction) * 100)
            predicted_category = class_labels[predicted_class]
            recommendation = get_recommendation(predicted_category)

            st.subheader("Prediction Result")
            if predicted_category == "Normal":
                st.success(f"Predicted Class: {predicted_category}")
            else:
                st.warning(f"Predicted Class: {predicted_category}")

            st.info(f"Confidence Score: {confidence:.2f}%")

            st.subheader("Recommendation")
            st.write(recommendation)




if __name__ == "__main__":
    insert()
