from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "crop_recommendation_pipeline.joblib"

FEATURES = [
    "N", "P", "K",
    "temperature", "humidity",
    "ph", "rainfall"
]

st.set_page_config(
    page_title="Agri-AI Crop Recommendation",
    page_icon="🌾",
    layout="centered",
)

st.title("🌾 Agri-AI Crop Recommendation")
st.write(
    "Enter soil and environmental conditions to get an AI-based crop recommendation."
)

if not MODEL_PATH.exists():
    st.error("Model not found. First run `python src/train.py`.")
    st.stop()

bundle = joblib.load(MODEL_PATH)
pipeline = bundle["pipeline"]

col1, col2 = st.columns(2)

with col1:
    N = st.number_input("Nitrogen (N)", min_value=0.0, value=90.0)
    P = st.number_input("Phosphorus (P)", min_value=0.0, value=42.0)
    K = st.number_input("Potassium (K)", min_value=0.0, value=43.0)
    temperature = st.number_input("Temperature (°C)", value=25.0)

with col2:
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=80.0)
    ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, value=200.0)

if st.button("🌱 Recommend Crop", type="primary", use_container_width=True):
    row = pd.DataFrame([{
        "N": N,
        "P": P,
        "K": K,
        "temperature": temperature,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall,
    }])

    prediction = pipeline.predict(row)[0]
    probabilities = pipeline.predict_proba(row)[0]
    classes = pipeline.named_steps["model"].classes_

    st.success(f"Recommended Crop: **{prediction.upper()}**")

    top_indices = probabilities.argsort()[::-1][:3]
    result = pd.DataFrame({
        "Crop": [classes[i] for i in top_indices],
        "Probability": [probabilities[i] for i in top_indices],
    })
    result["Probability"] = (result["Probability"] * 100).round(2).astype(str) + "%"

    st.subheader("Top 3 predictions")
    st.dataframe(result, use_container_width=True, hide_index=True)

st.caption("Educational project based on the Kaggle Crop Recommendation Dataset.")
