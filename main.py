import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import traceback

# =========================
# PAGE HEADER
# =========================
st.image("emphyscan.png", width=200)
st.header("Emphysema Chest X-ray Detector")

# =========================
# CLASS LABELS
# =========================
class_labels = {
    0: "Emphysema",
    1: "Normal",
}

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_emphysema_model():
    return load_model(
        "new_emphysema_model.keras",
        compile=False
    )

# =========================
# RECOMMENDATIONS
# =========================
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

    return "No recommendation available."


# =========================
# MAIN APP
# =========================
def insert():

    col1, col2 = st.columns([1, 1])

    # LEFT COLUMN
    with col1:
        uploaded_file = st.file_uploader(
            "",
            type=["jpg", "jpeg", "png", "bmp"],
            key="upl"
        )

        predict_btn = False

        if uploaded_file is None:
            st.warning("No image uploaded")

        else:
            img = Image.open(uploaded_file).convert("RGB")

            st.image(
                img,
                caption="Selected X-ray Image",
                use_container_width=True
            )

            predict_btn = st.button(
                "Predict",
                use_container_width=True
            )

    # RIGHT COLUMN
    with col2:

        if predict_btn:

            try:
                #st.subheader("Prediction Results")

                loaded_model = load_emphysema_model()

                img_array = img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)

                prediction = loaded_model.predict(img_array)

                predicted_class = int(
                    np.argmax(prediction, axis=1)[0]
                )

                confidence = float(
                    np.max(prediction) * 100
                )

                predicted_category = class_labels[predicted_class]

                recommendation = get_recommendation(
                    predicted_category
                )

                #st.subheader("Diagnosis")

                if predicted_category == "Normal":
                    st.success(
                        f"Predicted Class: {predicted_category}"
                    )
                else:
                    st.warning(
                        f"Predicted Class: {predicted_category}"
                    )

                st.metric(
                    "Confidence Score",
                    f"{confidence:.2f}%"
                )

                st.subheader("Recommendation")
                st.write(recommendation)

            except Exception:
                st.error("An error occurred during prediction.")
                st.code(traceback.format_exc())

        


# =========================
# RUN APP
# =========================
if __name__ == "__main__":
    insert()
