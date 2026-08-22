
import streamlit as st
import numpy as np
import tensorflow as tf
import pydicom
import cv2

MODEL_PATH = "best_pneumonia_model.keras"

model = tf.keras.models.load_model(MODEL_PATH)

st.title("Pneumonia Detection from Chest X-ray")

st.write(
    "Upload a chest X-ray image to predict whether "
    "pneumonia is present."
)

uploaded_file = st.file_uploader(
    "Upload X-ray",
    type=["dcm", "png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    file_name = uploaded_file.name.lower()

    if file_name.endswith(".dcm"):

        dicom = pydicom.dcmread(
            uploaded_file
        )

        image = dicom.pixel_array

    else:

        file_bytes = np.asarray(
            bytearray(uploaded_file.read()),
            dtype=np.uint8
        )

        image = cv2.imdecode(
            file_bytes,
            cv2.IMREAD_GRAYSCALE
        )

    image = image.astype(np.float32)

    image = (
        image - image.min()
    ) / (
        image.max() - image.min() + 1e-8
    )

    image = cv2.resize(
        image,
        (200, 200)
    )

    image = np.expand_dims(
        image,
        axis=-1
    )

    image = np.repeat(
        image,
        3,
        axis=-1
    )

    image = np.expand_dims(
        image,
        axis=0
    )

    prediction = model.predict(
        image,
        verbose=0
    )[0][0]

    if prediction >= 0.5:

        predicted_class = "Pneumonia"
        probability = prediction

    else:

        predicted_class = "No Pneumonia"
        probability = 1 - prediction

    st.image(
        image[0, :, :, 0],
        caption="Uploaded Chest X-ray",
        clamp=True
    )

    st.subheader(
        f"Prediction: {predicted_class}"
    )

    st.write(
        f"Probability: {probability * 100:.2f}%"
    )
