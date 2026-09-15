from pathlib import Path
import joblib
import pandas as pd
import streamlit as st


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "crop_recommendation_pipeline.joblib"

st.set_page_config(
    page_title="Agri-AI | Crop Recommendation",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# DARK THEME
# ============================================================

st.html("""
<style>

    /* ========================================================
       GLOBAL
    ======================================================== */

    html, body, [data-testid="stAppViewContainer"] {
        background: #0b0f0d !important;
    }

    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(
                circle at 85% 0%,
                rgba(50, 120, 75, 0.12),
                transparent 30%
            ),
            radial-gradient(
                circle at 0% 100%,
                rgba(39, 90, 57, 0.08),
                transparent 30%
            ),
            #0b0f0d !important;
    }

    [data-testid="stHeader"] {
        background: transparent !important;
    }

    [data-testid="stToolbar"] {
        display: none;
    }

    .main .block-container {
        max-width: 1250px;
        padding-top: 28px;
        padding-bottom: 70px;
    }


    /* ========================================================
       TOP NAVIGATION
    ======================================================== */

    .nav {
        height: 62px;
        display: flex;
        align-items: center;
        justify-content: space-between;

        border-bottom: 1px solid #202923;

        margin-bottom: 55px;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 12px;

        color: #f1f5f2;
        font-size: 18px;
        font-weight: 800;
    }

    .brand-icon {
        width: 38px;
        height: 38px;

        display: flex;
        align-items: center;
        justify-content: center;

        border-radius: 11px;

        background: #183c27;

        font-size: 20px;
    }

    .nav-right {
        color: #758178;
        font-size: 13px;
    }


    /* ========================================================
       HERO
    ======================================================== */

    .hero {
        margin-bottom: 48px;
    }

    .eyebrow {
        display: inline-block;

        padding: 7px 12px;

        border: 1px solid #244c32;
        border-radius: 20px;

        background: #101c15;

        color: #6fc284;

        font-size: 11px;
        font-weight: 800;

        letter-spacing: 1.2px;
        text-transform: uppercase;
    }

    .hero-title {
        margin-top: 18px;

        color: #f2f6f3;

        font-size: 48px;
        line-height: 1.05;

        font-weight: 850;

        letter-spacing: -1.8px;
    }

    .hero-title span {
        color: #62b976;
    }

    .hero-description {
        max-width: 680px;

        margin-top: 16px;

        color: #89958d;

        font-size: 15px;
        line-height: 1.7;
    }


    /* ========================================================
       SECTION HEADERS
    ======================================================== */

    .section-header {
        display: flex;
        align-items: center;
        gap: 12px;

        margin-bottom: 20px;
    }

    .section-icon {
        width: 40px;
        height: 40px;

        display: flex;
        align-items: center;
        justify-content: center;

        border-radius: 11px;

        background: #122117;

        border: 1px solid #203c2a;

        font-size: 19px;
    }

    .section-title {
        color: #e8eee9;

        font-size: 16px;
        font-weight: 800;
    }

    .section-subtitle {
        margin-top: 3px;

        color: #69756d;

        font-size: 12px;
    }


    /* ========================================================
       CARDS
    ======================================================== */

    [data-testid="stVerticalBlockBorderWrapper"] {
        background: #111714 !important;

        border: 1px solid #222d26 !important;

        border-radius: 18px !important;

        box-shadow:
            0 10px 35px rgba(0,0,0,0.22) !important;
    }


    /* ========================================================
       NUMBER INPUTS
    ======================================================== */

    [data-testid="stNumberInput"] label {
        color: #89968e !important;

        font-size: 12px !important;

        font-weight: 650 !important;
    }

    [data-testid="stNumberInput"] input {
        height: 43px !important;

        background: #0c110e !important;

        color: #e7eee9 !important;

        border: 1px solid #29352d !important;

        border-radius: 10px !important;

        font-size: 14px !important;

        font-weight: 700 !important;
    }

    [data-testid="stNumberInput"] input:focus {
        border-color: #4e9a62 !important;

        box-shadow:
            0 0 0 1px #4e9a62 !important;
    }

    [data-testid="stNumberInput"] button {
        background: #171f1a !important;

        color: #9aa69e !important;

        border: none !important;
    }


    /* ========================================================
       RECOMMEND BUTTON
    ======================================================== */

    div[data-testid="stFormSubmitButton"] button {
        height: 58px !important;

        border-radius: 13px !important;

        background: #2f8a4b !important;

        border: 1px solid #3b9d58 !important;

        color: white !important;

        font-size: 14px !important;

        font-weight: 800 !important;

        letter-spacing: 0.3px !important;

        box-shadow:
            0 10px 30px rgba(47,138,75,0.18) !important;

        transition: 0.2s ease !important;
    }

    div[data-testid="stFormSubmitButton"] button:hover {
        background: #399957 !important;

        border-color: #4aaa66 !important;

        transform: translateY(-1px);
    }


    /* ========================================================
       RESULT CARD
    ======================================================== */

    .result {
        margin-top: 35px;

        padding: 32px;

        border-radius: 20px;

        background:
            radial-gradient(
                circle at 90% 15%,
                rgba(79,160,96,0.15),
                transparent 35%
            ),
            #111a14;

        border: 1px solid #285438;

        box-shadow:
            0 15px 45px rgba(0,0,0,0.25);
    }

    .result-label {
        color: #6fc284;

        font-size: 11px;

        font-weight: 800;

        letter-spacing: 1.3px;

        text-transform: uppercase;
    }

    .result-row {
        display: flex;

        align-items: center;

        justify-content: space-between;

        gap: 30px;
    }

    .result-crop {
        margin-top: 10px;

        color: #f3f7f4;

        font-size: 42px;

        font-weight: 900;

        letter-spacing: -1px;

        text-transform: uppercase;
    }

    .result-text {
        margin-top: 7px;

        color: #84928a;

        font-size: 13px;
    }

    .confidence {
        min-width: 170px;

        padding: 18px;

        border-radius: 14px;

        background: #0c120e;

        border: 1px solid #24372a;

        text-align: center;
    }

    .confidence-label {
        color: #69766d;

        font-size: 10px;

        font-weight: 800;

        letter-spacing: 1px;

        text-transform: uppercase;
    }

    .confidence-value {
        margin-top: 5px;

        color: #70c381;

        font-size: 25px;

        font-weight: 850;
    }


    /* ========================================================
       TOP PREDICTIONS
    ======================================================== */

    .predictions-heading {
        margin-top: 42px;

        color: #e9efeb;

        font-size: 20px;

        font-weight: 850;
    }

    .predictions-subheading {
        margin-top: 5px;

        margin-bottom: 18px;

        color: #68746c;

        font-size: 12px;
    }

    .prediction-card {
        min-height: 145px;

        padding: 20px;

        border-radius: 16px;

        background: #111714;

        border: 1px solid #222d26;

        box-shadow:
            0 8px 25px rgba(0,0,0,0.18);
    }

    .rank {
        color: #69756d;

        font-size: 10px;

        font-weight: 800;

        letter-spacing: 1px;

        text-transform: uppercase;
    }

    .crop {
        margin-top: 8px;

        color: #edf3ef;

        font-size: 18px;

        font-weight: 850;

        text-transform: capitalize;
    }

    .probability {
        margin-top: 5px;

        color: #66ba78;

        font-size: 13px;

        font-weight: 800;
    }

    .bar {
        height: 5px;

        margin-top: 17px;

        overflow: hidden;

        border-radius: 20px;

        background: #202a23;
    }

    .bar-fill {
        height: 100%;

        border-radius: 20px;

        background: #3e9655;
    }


    /* ========================================================
       SUMMARY
    ======================================================== */

    .summary {
        margin-top: 25px;

        padding: 15px 18px;

        border-radius: 12px;

        background: #0f1712;

        border: 1px solid #202d24;

        color: #77837b;

        font-size: 11px;

        line-height: 1.8;
    }

    .summary strong {
        color: #aab5ae;
    }


    /* ========================================================
       FOOTER
    ======================================================== */

    .footer {
        margin-top: 60px;

        padding-top: 22px;

        border-top: 1px solid #202923;

        text-align: center;

        color: #515d55;

        font-size: 11px;
    }


    /* ========================================================
       MOBILE
    ======================================================== */

    @media (max-width: 700px) {

        .hero-title {
            font-size: 35px;
        }

        .nav-right {
            display: none;
        }

        .result-row {
            flex-direction: column;

            align-items: flex-start;
        }

        .confidence {
            width: 100%;
        }

    }

</style>
""")


# ============================================================
# LOAD MODEL
# ============================================================

if not MODEL_PATH.exists():
    st.error("Model not found. First run `python src/train.py`.")
    st.stop()

bundle = joblib.load(MODEL_PATH)

pipeline = bundle["pipeline"]


# ============================================================
# NAVIGATION
# ============================================================

st.html("""
<div class="nav">

    <div class="brand">
        <div class="brand-icon">🌾</div>
        <div>Agri-AI</div>
    </div>

   
</div>
""")


# ============================================================
# HERO
# ============================================================

st.html("""
<div class="hero">

    
    <div class="hero-title">
        Find the right crop<br>
        <span>for your soil.</span>
    </div>

    <div class="hero-description">
        Enter your soil and environmental conditions.
        Agri-AI analyzes these parameters using a machine
        learning model and recommends the most suitable crop.
    </div>

</div>
""")


# ============================================================
# INPUT FORM
# ============================================================

with st.form("crop_form"):

    left, right = st.columns(2, gap="large")


    # --------------------------------------------------------
    # SOIL
    # --------------------------------------------------------

    with left:

        with st.container(border=True):

            st.html("""
            <div class="section-header">

                <div class="section-icon">
                    🌱
                </div>

                <div>
                    <div class="section-title">
                        Soil Conditions
                    </div>

                    <div class="section-subtitle">
                        Nutrient and acidity levels
                    </div>
                </div>

            </div>
            """)

            a, b = st.columns(2)

            with a:

                N = st.number_input(
                    "Nitrogen (N)",
                    min_value=0.0,
                    value=90.0,
                    step=1.0,
                )

                P = st.number_input(
                    "Phosphorus (P)",
                    min_value=0.0,
                    value=42.0,
                    step=1.0,
                )

            with b:

                K = st.number_input(
                    "Potassium (K)",
                    min_value=0.0,
                    value=43.0,
                    step=1.0,
                )

                ph = st.number_input(
                    "Soil pH",
                    min_value=0.0,
                    max_value=14.0,
                    value=6.5,
                    step=0.1,
                )


    # --------------------------------------------------------
    # ENVIRONMENT
    # --------------------------------------------------------

    with right:

        with st.container(border=True):

            st.html("""
            <div class="section-header">

                <div class="section-icon">
                    🌦️
                </div>

                <div>
                    <div class="section-title">
                        Environmental Conditions
                    </div>

                    <div class="section-subtitle">
                        Climate and rainfall parameters
                    </div>
                </div>

            </div>
            """)

            a, b = st.columns(2)

            with a:

                temperature = st.number_input(
                    "Temperature (°C)",
                    value=25.0,
                    step=0.5,
                )

                humidity = st.number_input(
                    "Humidity (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=80.0,
                    step=1.0,
                )

            with b:

                rainfall = st.number_input(
                    "Rainfall (mm)",
                    min_value=0.0,
                    value=200.0,
                    step=1.0,
                )

                st.html("""
                <div style="
                    color:#647168;
                    font-size:11px;
                    line-height:1.5;
                    margin-top:28px;
                ">
                    🌦️ Climate parameters help determine
                    which crops are best suited to the
                    selected environment.
                </div>
                """)


    st.write("")

    submitted = st.form_submit_button(
        "🌾  ANALYZE FARM & RECOMMEND CROP",
        use_container_width=True,
    )


# ============================================================
# PREDICTION
# ============================================================

if submitted:

    row = pd.DataFrame(
        [{
            "N": N,
            "P": P,
            "K": K,
            "temperature": temperature,
            "humidity": humidity,
            "ph": ph,
            "rainfall": rainfall,
        }]
    )

    prediction = pipeline.predict(row)[0]

    probabilities = pipeline.predict_proba(row)[0]

    classes = pipeline.named_steps["model"].classes_

    top_indices = probabilities.argsort()[::-1][:3]

    top_predictions = []

    for index in top_indices:

        top_predictions.append({
            "crop": classes[index],
            "probability": probabilities[index] * 100,
        })


    best_probability = top_predictions[0]["probability"]


    # ========================================================
    # RESULT
    # ========================================================

    st.html(f"""
    <div class="result">

        <div class="result-row">

            <div>

                <div class="result-label">
                    ✦ AI Recommendation
                </div>

                <div class="result-crop">
                    🌾 {prediction}
                </div>

                <div class="result-text">
                    Highest probability crop based on your
                    soil and environmental conditions.
                </div>

            </div>

            <div class="confidence">

                <div class="confidence-label">
                    Model Confidence
                </div>

                <div class="confidence-value">
                    {best_probability:.2f}%
                </div>

            </div>

        </div>

    </div>
    """)


    # ========================================================
    # TOP 3
    # ========================================================

    st.html("""
    <div class="predictions-heading">
        Top 3 Recommendations
    </div>

    <div class="predictions-subheading">
        Alternative crops ranked by model probability
    </div>
    """)


    columns = st.columns(3, gap="medium")

    medals = ["🥇", "🥈", "🥉"]

    for i, (column, item) in enumerate(
        zip(columns, top_predictions)
    ):

        with column:

            probability = item["probability"]

            width = min(probability, 100)

            st.html(f"""
            <div class="prediction-card">

                <div class="rank">
                    {medals[i]} &nbsp; RANK {i + 1}
                </div>

                <div class="crop">
                    {item["crop"]}
                </div>

                <div class="probability">
                    {probability:.2f}%
                </div>

                <div class="bar">
                    <div
                        class="bar-fill"
                        style="width:{width}%"
                    ></div>
                </div>

            </div>
            """)


    # ========================================================
    # INPUT SUMMARY
    # ========================================================

    st.html(f"""
    <div class="summary">

        <strong>Analysis parameters:</strong>

        &nbsp; N {N:.1f}
        &nbsp;•&nbsp; P {P:.1f}
        &nbsp;•&nbsp; K {K:.1f}
        &nbsp;•&nbsp; pH {ph:.1f}
        &nbsp;•&nbsp; {temperature:.1f}°C
        &nbsp;•&nbsp; {humidity:.1f}% humidity
        &nbsp;•&nbsp; {rainfall:.1f} mm rainfall

    </div>
    """)


# ============================================================
# FOOTER
# ============================================================

st.html("""
<div class="footer">

    🌾 Agri-AI Crop Recommendation
    

</div>
""")