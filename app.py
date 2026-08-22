import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

from tensorflow.keras.applications.efficientnet import preprocess_input


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="🫁",
    layout="centered"
)


# --------------------------------------------------
# Load trained model
# --------------------------------------------------

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        "best_pneumonia_model.keras",
        custom_objects={
            "preprocess_input": preprocess_input
        },
        compile=False,
        safe_mode=False
    )

    return model


model = load_model()


# --------------------------------------------------
# Application title
# --------------------------------------------------

st.title("🫁 Pneumonia Detection")

st.write(
    "Upload a chest X-ray image to obtain a pneumonia prediction."
)


# --------------------------------------------------
# Upload image
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Chest X-ray",
    type=["jpg", "jpeg", "png"]
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file)

    # Display original image
    st.subheader("Uploaded X-ray")

    st.image(
        image,
        use_container_width=True
    )

    # ----------------------------------------------
    # Convert to grayscale
    # ----------------------------------------------

    image_gray = image.convert("L")

    # ----------------------------------------------
    # Resize to model input size
    # ----------------------------------------------

    image_gray = image_gray.resize((200, 200))

    # ----------------------------------------------
    # Convert to NumPy array
    # ----------------------------------------------

    image_array = np.array(image_gray)

    # ----------------------------------------------
    # Add channel dimension
    # ----------------------------------------------

    image_array = np.expand_dims(
        image_array,
        axis=-1
    )

    # ----------------------------------------------
    # Repeat grayscale channel 3 times
    # Same preprocessing used during training
    # ----------------------------------------------

    image_rgb = np.repeat(
        image_array,
        3,
        axis=-1
    )

    # ----------------------------------------------
    # Add batch dimension
    # ----------------------------------------------

    image_rgb = np.expand_dims(
        image_rgb,
        axis=0
    )

    # ----------------------------------------------
    # Prediction
    # ----------------------------------------------

    prediction = model.predict(
        image_rgb,
        verbose=0
    )[0][0]

    # ----------------------------------------------
    # Classification
    # ----------------------------------------------

    if prediction >= 0.5:

        predicted_class = "Pneumonia"
        probability = prediction

    else:

        predicted_class = "No Pneumonia"
        probability = 1 - prediction


    # ----------------------------------------------
    # Display result
    # ----------------------------------------------

    st.subheader("Prediction")

    if predicted_class == "Pneumonia":

        st.error(
            f"Prediction: {predicted_class}"
        )

    else:

        st.success(
            f"Prediction: {predicted_class}"
        )


    st.write(
        f"Probability: {probability * 100:.2f}%"
    )