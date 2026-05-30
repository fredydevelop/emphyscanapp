import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

class_labels = {0: "Emphysema", 1: "Normal"}

@st.cache_resource
def load_emphy_model():
    return load_model("main_emphysema_model.keras", compile=False)

model = load_emphy_model()

st.title("Emphysema Chest Xray Detector")

uploaded_file = st.file_uploader("Upload image", type=["jpg","jpeg","png","bmp"])

predict = st.button("Predict")

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, width=400)

    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    if predict:
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        st.write(class_labels[predicted_class])
