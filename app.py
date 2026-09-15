from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "crop_recommendation_pipeline.joblib"

FEATURES = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall",
]

st.set_page_config(
    page_title="Agri-AI | Smart Crop Recommendation",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ---------- GLOBAL ---------- */

    .stApp {
        background:
            radial-gradient(
                circle at 85% 5%,
                rgba(42, 157, 96, 0.10),
                transparent 28%
            ),
            radial-gradient(
                circle at 5% 90%,
                rgba(230, 180, 70, 0.07),
                transparent 25%
            ),
            #f7f9f6;
    }

    .main .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    /* Remove Streamlit top padding */
    [data-testid="stHeader"] {
        background: transparent;
    }

    /* ---------- TOP NAV ---------- */

    .top-nav {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.4rem 0 1.8rem 0;
        border-bottom: 1px solid #e4e9e3;
        margin-bottom: 3rem;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 1.15rem;
        font-weight: 800;
        color: #173b28;
        letter-spacing: -0.3px;
    }

    .brand-icon {
        width: 38px;
        height: 38px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: #173b28;
        font-size: 21px;
    }

    .nav-right {
        color: #718075;
        font-size: 0.88rem;
    }

    /* ---------- HERO ---------- */

    .hero {
        padding: 0 0 2.8rem 0;
    }

    .hero-tag {
        display: inline-block;
        padding: 7px 13px;
        border-radius: 50px;
        background: #e7f3e9;
        color: #267044;
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.7px;
        margin-bottom: 1rem;
        text-transform: uppercase;
    }

    .hero h1 {
        font-size: 3.15rem;
        line-height: 1.08;
        letter-spacing: -2px;
        color: #173b28;
        margin: 0;
        font-weight: 850;
    }

    .hero h1 span {
        color: #3d8b54;
    }

    .hero p {
        color: #68756c;
        font-size: 1.05rem;
        margin-top: 1rem;
        max-width: 650px;
        line-height: 1.65;
    }

    /* ---------- SECTION HEADINGS ---------- */

    .section-heading {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 1rem 0 1rem 0;
    }

    .section-icon {
        width: 38px;
        height: 38px;
        border-radius: 11px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: #e8f3e9;
        font-size: 19px;
    }

    .section-title {
        color: #173b28;
        font-size: 1.15rem;
        font-weight: 800;
        margin: 0;
    }

    .section-subtitle {
        color: #7b867d;
        font-size: 0.78rem;
        margin-top: 2px;
    }

    /* ---------- INPUT CARDS ---------- */

    [data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255, 255, 255, 0.82);
        border: 1px solid #e1e8e1;
        border-radius: 20px;
        box-shadow: 0 8px 30px rgba(31, 62, 42, 0.055);
    }

    /* Input labels */
    [data-testid="stNumberInput"] label {
        color: #46554b !important;
        font-weight: 700 !important;
        font-size: 0.86rem !important;
    }

    /* Input box */
    [data-testid="stNumberInput"] input {
        background: #f8faf8 !important;
        border: 1px solid #dfe6df !important;
        border-radius: 11px !important;
        color: #1c3324 !important;
        font-weight: 650 !important;
    }

    [data-testid="stNumberInput"] input:focus {
        border-color: #4d9361 !important;
        box-shadow: 0 0 0 2px rgba(77, 147, 97, 0.12) !important;
    }

    /* ---------- BUTTON ---------- */

    div[data-testid="stFormSubmitButton"] button {
        width: 100%;
        min-height: 58px;
        border-radius: 14px;
        border: none;
        background: #173b28;
        color: white;
        font-size: 1rem;
        font-weight: 800;
        letter-spacing: 0.2px;
        box-shadow: 0 10px 25px rgba(23, 59, 40, 0.18);
        transition: all 0.2s ease;
    }

    div[data-testid="stFormSubmitButton"] button:hover {
        background: #245c3b;
        transform: translateY(-2px);
        box-shadow: 0 14px 30px rgba(23, 59, 40, 0.23);
    }

    /* ---------- RESULT HERO ---------- */

    .result-card {
        position: relative;
        overflow: hidden;
        margin-top: 2.5rem;
        padding: 2.4rem;
        border-radius: 24px;
        background:
            radial-gradient(
                circle at 90% 20%,
                rgba(126, 190, 108, 0.20),
                transparent 30%
            ),
            linear-gradient(135deg, #173b28, #245c3b);
        color: white;
        box-shadow: 0 18px 45px rgba(23, 59, 40, 0.20);
    }

    .result-label {
        font-size: 0.76rem;
        font-weight: 800;
        letter-spacing: 1.3px;
        text-transform: uppercase;
        color: #a9d5b0;
    }

    .result-crop {
        font-size: 3rem;
        line-height: 1;
        font-weight: 900;
        letter-spacing: -1px;
        margin-top: 0.7rem;
        text-transform: uppercase;
    }

    .result-description {
        color: #d4e5d7;
        margin-top: 0.7rem;
        font-size: 0.95rem;
    }

    .confidence-box {
        margin-top: 1.7rem;
        padding-top: 1.2rem;
        border-top: 1px solid rgba(255,255,255,0.15);
    }

    .confidence-title {
        color: #b7cfbb;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }

    .confidence-value {
        font-size: 1.65rem;
        font-weight: 850;
        margin-top: 2px;
    }

    /* ---------- TOP PREDICTIONS ---------- */

    .predictions-title {
        margin-top: 2.8rem;
        color: #173b28;
        font-size: 1.35rem;
        font-weight: 850;
    }

    .predictions-subtitle {
        color: #7b867d;
        font-size: 0.85rem;
        margin-bottom: 1.2rem;
    }

    .prediction-card {
        background: white;
        border: 1px solid #e1e8e1;
        border-radius: 18px;
        padding: 1.35rem;
        min-height: 145px;
        box-shadow: 0 7px 25px rgba(31, 62, 42, 0.045);
    }

    .prediction-rank {
        color: #8a958d;
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.8px;
        text-transform: uppercase;
    }

    .prediction-crop {
        color: #173b28;
        font-size: 1.25rem;
        font-weight: 850;
        margin-top: 7px;
        text-transform: capitalize;
    }

    .prediction-percent {
        color: #3d8b54;
        font-size: 1rem;
        font-weight: 800;
        margin-top: 5px;
    }

    .progress-bg {
        height: 7px;
        width: 100%;
        background: #edf1ed;
        border-radius: 20px;
        overflow: hidden;
        margin-top: 14px;
    }

    .progress-fill {
        height: 100%;
        background: #4d9361;
        border-radius: 20px;
    }

    /* ---------- INFO STRIP ---------- */

    .info-strip {
        margin-top: 2.5rem;
        padding: 1rem 1.3rem;
        border-radius: 14px;
        background: #edf5ee;
        border: 1px solid #dce9dd;
        color: #49604f;
        font-size: 0.82rem;
    }

    /* ---------- FOOTER ---------- */

    .footer {
        margin-top: 4rem;
        padding-top: 1.5rem;
        border-top: 1px solid #e1e7e1;
        text-align: center;
        color: #879188;
        font-size: 0.76rem;
    }

    /* ---------- MOBILE ---------- */

    @media (max-width: 768px) {

        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .hero h1 {
            font-size: 2.25rem;
        }

        .result-crop {
            font-size: 2.3rem;
        }

        .top-nav {
            margin-bottom: 2rem;
        }

        .nav-right {
            display: none;
        }
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD MODEL
# ============================================================

if not MODEL_PATH.exists():
    st.error("Model not found. First run `python src/train.py`.")
    st.stop()

bundle = joblib.load(MODEL_PATH)

pipeline = bundle["pipeline"]


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="top-nav">
        <div class="brand">
            <div class="brand-icon">🌾</div>
            <div>Agri-AI</div>
        </div>

        <div class="nav-right">
            Smart Agriculture • Machine Learning
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-tag">AI-Powered Agriculture</div>

        <h1>
            Find the right crop<br>
            <span>for your soil.</span>
        </h1>

        <p>
            Enter your soil and environmental conditions.
            Our machine learning model analyzes the inputs and
            recommends the most suitable crop for cultivation.
        </p>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# INPUT FORM
# ============================================================

with st.form("crop_recommendation_form"):

    col1, col2 = st.columns(2, gap="large")

    # --------------------------------------------------------
    # SOIL CARD
    # --------------------------------------------------------

    with col1:

        with st.container(border=True):

            st.markdown(
                """
                <div class="section-heading">

                    <div class="section-icon">🌱</div>

                    <div>
                        <div class="section-title">
                            Soil Conditions
                        </div>

                        <div class="section-subtitle">
                            Nutrient and acidity levels
                        </div>
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

            soil_a, soil_b = st.columns(2)

            with soil_a:
                N = st.number_input(
                    "Nitrogen (N)",
                    min_value=0.0,
                    value=90.0,
                    step=1.0,
                    help="Nitrogen content in the soil.",
                )

                P = st.number_input(
                    "Phosphorus (P)",
                    min_value=0.0,
                    value=42.0,
                    step=1.0,
                    help="Phosphorus content in the soil.",
                )

            with soil_b:
                K = st.number_input(
                    "Potassium (K)",
                    min_value=0.0,
                    value=43.0,
                    step=1.0,
                    help="Potassium content in the soil.",
                )

                ph = st.number_input(
                    "Soil pH",
                    min_value=0.0,
                    max_value=14.0,
                    value=6.5,
                    step=0.1,
                    help="Soil acidity / alkalinity.",
                )


    # --------------------------------------------------------
    # ENVIRONMENT CARD
    # --------------------------------------------------------

    with col2:

        with st.container(border=True):

            st.markdown(
                """
                <div class="section-heading">

                    <div class="section-icon">☁️</div>

                    <div>
                        <div class="section-title">
                            Environmental Conditions
                        </div>

                        <div class="section-subtitle">
                            Climate and rainfall parameters
                        </div>
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

            env_a, env_b = st.columns(2)

            with env_a:

                temperature = st.number_input(
                    "Temperature (°C)",
                    value=25.0,
                    step=0.5,
                    help="Average temperature in degrees Celsius.",
                )

                humidity = st.number_input(
                    "Humidity (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=80.0,
                    step=1.0,
                    help="Relative humidity percentage.",
                )

            with env_b:

                rainfall = st.number_input(
                    "Rainfall (mm)",
                    min_value=0.0,
                    value=200.0,
                    step=1.0,
                    help="Expected rainfall in millimeters.",
                )

                st.markdown(
                    """
                    <div style="
                        height: 100%;
                        min-height: 65px;
                        display:flex;
                        align-items:center;
                        color:#7b867d;
                        font-size:0.78rem;
                        padding:0.5rem;
                    ">
                        🌦️ Climate conditions help determine
                        which crops are naturally suited to
                        the selected environment.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

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
        [
            {
                "N": N,
                "P": P,
                "K": K,
                "temperature": temperature,
                "humidity": humidity,
                "ph": ph,
                "rainfall": rainfall,
            }
        ]
    )

    # Prediction
    prediction = pipeline.predict(row)[0]

    # Probabilities
    probabilities = pipeline.predict_proba(row)[0]

    # Model classes
    classes = pipeline.named_steps["model"].classes_

    # Top 3
    top_indices = probabilities.argsort()[::-1][:3]

    top_crops = [
        {
            "crop": classes[i],
            "probability": float(probabilities[i] * 100),
        }
        for i in top_indices
    ]

    best_probability = top_crops[0]["probability"]


    # ========================================================
    # MAIN RESULT
    # ========================================================

    st.markdown(
        f"""
        <div class="result-card">

            <div class="result-label">
                ✦ AI Recommendation
            </div>

            <div class="result-crop">
                🌾 {prediction}
            </div>

            <div class="result-description">
                This is the highest-probability crop based on
                the soil and environmental conditions you provided.
            </div>

            <div class="confidence-box">

                <div class="confidence-title">
                    Model confidence
                </div>

                <div class="confidence-value">
                    {best_probability:.2f}%
                </div>

            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


    # ========================================================
    # TOP 3 PREDICTIONS
    # ========================================================

    st.markdown(
        """
        <div class="predictions-title">
            Top 3 Recommendations
        </div>

        <div class="predictions-subtitle">
            Alternative crops ranked by model probability
        </div>
        """,
        unsafe_allow_html=True,
    )

    prediction_columns = st.columns(3, gap="medium")

    medals = ["🥇", "🥈", "🥉"]

    for index, (column, item) in enumerate(
        zip(prediction_columns, top_crops)
    ):

        with column:

            probability = item["probability"]

            # Keep progress bar visually within 100%
            progress_width = min(probability, 100)

            st.markdown(
                f"""
                <div class="prediction-card">

                    <div class="prediction-rank">
                        {medals[index]} Rank {index + 1}
                    </div>

                    <div class="prediction-crop">
                        {item["crop"]}
                    </div>

                    <div class="prediction-percent">
                        {probability:.2f}%
                    </div>

                    <div class="progress-bg">
                        <div
                            class="progress-fill"
                            style="width:{progress_width}%"
                        ></div>
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )


    # ========================================================
    # INPUT SUMMARY
    # ========================================================

    st.markdown(
        f"""
        <div class="info-strip">

            <strong>📊 Analysis summary:</strong>

            N {N:.1f} &nbsp;•&nbsp;
            P {P:.1f} &nbsp;•&nbsp;
            K {K:.1f} &nbsp;•&nbsp;
            pH {ph:.1f} &nbsp;•&nbsp;
            {temperature:.1f}°C &nbsp;•&nbsp;
            {humidity:.1f}% humidity &nbsp;•&nbsp;
            {rainfall:.1f} mm rainfall

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

        🌾 Agri-AI Crop Recommendation &nbsp;•&nbsp;
        Powered by Machine Learning &nbsp;•&nbsp;
        Random Forest

    </div>
    """,
    unsafe_allow_html=True,
)