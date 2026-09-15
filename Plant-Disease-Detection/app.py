import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="centered"
)

# -----------------------------
# Dark UI
# -----------------------------
st.markdown("""
<style>
    .stApp {
        background-color: #0b0f0d;
        color: #f5f5f5;
    }

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #a8b3ad;
        font-size: 17px;
        margin-bottom: 35px;
    }

    .result-box {
        background-color: #151b18;
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        margin-top: 25px;
        border: 1px solid #29332e;
    }

    .disease {
        font-size: 27px;
        font-weight: 600;
        margin: 10px;
    }

    .confidence {
        font-size: 18px;
        color: #b8c4bd;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Title
# -----------------------------
st.markdown(
    '<div class="main-title">🌿 Plant Disease Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Upload a plant leaf image to detect its disease using CNN</div>',
    unsafe_allow_html=True
)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/plant_disease_cnn.keras")

model = load_model()

# -----------------------------
# Class Names
# -----------------------------
class_names = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy"
]

# -----------------------------
# Upload Image
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload a plant leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Leaf Image",
        use_container_width=True
    )

    # -----------------------------
    # Prediction
    # -----------------------------
    image_resized = image.resize((128, 128))

    image_array = np.array(image_resized) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    prediction = model.predict(image_array, verbose=0)

    predicted_index = np.argmax(prediction[0])
    confidence = prediction[0][predicted_index] * 100

    predicted_class = class_names[predicted_index]

    # -----------------------------
    # Display Result
    # -----------------------------
    st.markdown(
        f"""
        <div class="result-box">
            <div>Predicted Disease</div>
            <div class="disease">{predicted_class}</div>
            <div class="confidence">
                Confidence: {confidence:.2f}%
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )