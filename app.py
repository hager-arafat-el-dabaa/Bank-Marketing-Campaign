
import os
import streamlit as st
import pandas as pd
import joblib


# =========================================================python -m streamlit run app.py
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="BANKAI | Customer Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# LOAD MODELS
# =========================================================

@st.cache_resource
def load_models():

    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")

    rf_path = os.path.join(models_dir, "rf_model.pkl")
    lgbm_path = os.path.join(models_dir, "lgbm_model.pkl")
    scaler_path = os.path.join(models_dir, "scaler.pkl")

    if not os.path.exists(rf_path):
        raise FileNotFoundError(
            f"Random Forest model not found:\n{rf_path}"
        )

    if not os.path.exists(lgbm_path):
        raise FileNotFoundError(
            f"LightGBM model not found:\n{lgbm_path}"
        )

    if not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"Scaler not found:\n{scaler_path}"
        )

    rf_model = joblib.load(rf_path)
    lgbm_model = joblib.load(lgbm_path)
    scaler = joblib.load(scaler_path)

    return rf_model, lgbm_model, scaler


rf_model, lgbm_model, scaler = load_models()


# =========================================================
# CSS ONLY
# =========================================================

st.markdown(
    """
    <style>

    /* =========================
       COLORS
       ========================= */

    :root {
        --olive: #68743A;
        --dark-olive: #3F4825;
        --light-olive: #EEF1E4;
        --border: #D9DEC9;
        --text: #303522;
        --muted: #777C6B;
        --white: #FFFFFF;
    }


    /* =========================
       PAGE
       ========================= */

    .stApp {
        background: #FFFFFF;
    }

    .block-container {
        max-width: 1450px;
        padding: 45px 60px 70px 60px;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }


    /* =========================
       HERO
       ========================= */

    .hero-small {
        color: #68743A !important;
        font-size: 14px !important;
        font-weight: 800 !important;
        letter-spacing: 3px !important;
        margin-bottom: 18px !important;
    }

    .hero-title {
        color: #3F4825 !important;
        font-size: 58px !important;
        font-weight: 800 !important;
        line-height: 1.08 !important;
        margin-bottom: 20px !important;
    }

    .hero-green {
        color: #68743A !important;
    }

    .hero-description {
        color: #777C6B !important;
        font-size: 17px !important;
        line-height: 1.7 !important;
        max-width: 850px !important;
        margin-bottom: 40px !important;
    }


    /* =========================
       SECTION TITLES
       ========================= */

    .section-title {
        color: #3F4825 !important;
        font-size: 27px !important;
        font-weight: 800 !important;
        margin-top: 30px !important;
        margin-bottom: 5px !important;
    }

    .section-subtitle {
        color: #777C6B !important;
        font-size: 14px !important;
        margin-bottom: 25px !important;
    }


    /* =========================
       LABELS
       ========================= */

    
       

    label {
      color: #68743A !important;
      font-size: 15px !important;
      font-weight: 700 !important;
    }

    /* =========================
       INPUTS
       ========================= */

    div[data-baseweb="input"] > div {
        background: #FFFFFF !important;
        border: 1px solid #D9DEC9 !important;
        border-radius: 13px !important;
        min-height: 62px !important;
        padding: 8px 15px !important;
        box-shadow: 0 4px 14px rgba(63, 72, 37, 0.04) !important;
    }

    div[data-baseweb="input"] input {
        color: #303522 !important;
        font-size: 16px !important;
    }


    /* =========================
       SELECT BOXES
       ========================= */

    div[data-baseweb="select"] > div {
        background: #FFFFFF !important;
        border: 1px solid #D9DEC9 !important;
        border-radius: 13px !important;
        min-height: 62px !important;
        padding: 8px 15px !important;
        box-shadow: 0 4px 14px rgba(63, 72, 37, 0.04) !important;
    }


    /* =========================
       FOCUS
       ========================= */

    div[data-baseweb="input"] > div:focus-within {
        border: 2px solid #68743A !important;
    }

    div[data-baseweb="select"] > div:focus-within {
        border: 2px solid #68743A !important;
    }


    /* =========================
       MODEL BOX
       ========================= */

    .model-box {
        background: #FFFFFF;
        border: 1px solid #D9DEC9;
        border-radius: 13px;
        min-height: 62px;
        padding: 14px 18px;
        margin-top: 28px;
        box-shadow: 0 4px 14px rgba(63, 72, 37, 0.04);
    }

    .model-title {
        color: #68743A;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 1.5px;
    }

    .model-text {
        color: #3F4825;
        font-size: 15px;
        font-weight: 700;
        margin-top: 5px;
        line-height: 1.5;
    }


    /* =========================
       BUTTON
       ========================= */

    div.stButton > button {
        width: 100% !important;
        min-height: 68px !important;
        background: #68743A !important;
        color:#3F4825; !important;
        border: none !important;
        border-radius: 14px !important;
        font-size: 18px !important;
        font-weight: 800 !important;
        box-shadow: 0 8px 20px rgba(63, 72, 37, 0.15) !important;
    }

    div.stButton > button:hover {
        background:#3F4825; !important;
        color: #FFFFFF !important;
    }


    /* =========================
       RESULT
       ========================= */

    .result-box {
        background: #3F4825;
        border-radius: 20px;
        padding: 35px;
        margin-top: 30px;
        margin-bottom: 25px;
        box-shadow: 0 12px 30px rgba(63, 72, 37, 0.18);
    }

    .result-label {
        color: #3F4825;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 2px;
    }

    .result-title {
        color:#3F4825;
        font-size: 32px;
        font-weight: 800;
        margin-top: 10px;
    }

    .result-probability {
        color:#3F4825;
        font-size: 17px;
        margin-top: 8px;
    }


    /* =========================
       METRICS
       ========================= */

    div[data-testid="stMetric"] {
        background: #FFFFFF !important;
        border: 1px solid #D9DEC9 !important;
        border-radius: 13px !important;
        min-height: 105px !important;
        padding: 20px 22px !important;
        box-shadow: 0 5px 18px rgba(63, 72, 37, 0.05) !important;
    }

    div[data-testid="stMetricLabel"] {
        color: #303522 !important;
        font-size: 14px !important;
        font-weight: 700 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #3F4825 !important;
        font-size: 28px !important;
        font-weight: 800 !important;
    }


    /* =========================
       INSIGHTS
       ========================= */

    .insight-box {
        background: #FFFFFF;
        border: 1px solid #D9DEC9;
        border-radius: 15px;
        padding: 22px;
        min-height: 120px;
        box-shadow: 0 5px 18px rgba(63, 72, 37, 0.05);
    }

    .insight-title {
        color: #3F4825;
        font-size: 16px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .insight-text {
        color: #777C6B;
        font-size: 14px;
        line-height: 1.6;
    }


    /* =========================
       FOOTER
       ========================= */

    .footer {
        text-align: center;
        color: #777C6B;
        margin-top: 55px;
        font-size: 13px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HERO
# =========================================================

st.markdown(
    '<div class="hero-small">BANKAI · CUSTOMER INTELLIGENCE</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="hero-title">
        Bank Marketing Campaign
    </div>
    """,
    unsafe_allow_html=True
)

# =========================================================
# CUSTOMER PROFILE
# =========================================================
st.markdown(
    '<div class="section-title">Customer Profile</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">'
    'Enter the customer\'s personal and financial information.'
    '</div>',
    unsafe_allow_html=True
)


col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=35
    )

with col2:
    job = st.selectbox(
        "Job",
        [
            "admin.",
            "blue-collar",
            "entrepreneur",
            "housemaid",
            "management",
            "retired",
            "self-employed",
            "services",
            "student",
            "technician",
            "unemployed",
            "unknown"
        ]
    )

with col3:
    marital = st.selectbox(
        "Marital Status",
        [
            "married",
            "single",
            "divorced"
        ]
    )


col4, col5, col6 = st.columns(3)

with col4:
    education = st.selectbox(
        "Education",
        [
            "primary",
            "secondary",
            "tertiary",
            "unknown"
        ]
    )

with col5:
    default = st.selectbox(
        "Credit Default",
        [
            "no",
            "yes"
        ]
    )

with col6:
    balance = st.number_input(
        "Account Balance",
        value=1000
    )


col7, col8, col9 = st.columns(3)

with col7:
    housing = st.selectbox(
        "Housing Loan",
        [
            "no",
            "yes"
        ]
    )

with col8:
    loan = st.selectbox(
        "Personal Loan",
        [
            "no",
            "yes"
        ]
    )

with col9:
    contact = st.selectbox(
        "Contact Type",
        [
            "cellular",
            "telephone",
            "unknown"
        ]
    )


# =========================================================
# MARKETING CAMPAIGN
# =========================================================

st.markdown(
    '<div class="section-title">Marketing Campaign</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">'
    'Provide information about the current and previous campaign activity.'
    '</div>',
    unsafe_allow_html=True
)


col1, col2, col3 = st.columns(3)

with col1:
    day = st.number_input(
        "Contact Day",
        min_value=1,
        max_value=31,
        value=15
    )

with col2:
    month = st.selectbox(
        "Contact Month",
        [
            "jan",
            "feb",
            "mar",
            "apr",
            "may",
            "jun",
            "jul",
            "aug",
            "sep",
            "oct",
            "nov",
            "dec"
        ]
    )

with col3:
    duration = st.number_input(
        "Call Duration (seconds)",
        min_value=0,
        max_value=5000,
        value=300
    )


col4, col5, col6 = st.columns(3)

with col4:
    campaign = st.number_input(
        "Campaign Contacts",
        min_value=1,
        max_value=100,
        value=1
    )

with col5:
    pdays = st.number_input(
        "Days Since Previous Contact",
        min_value=-1,
        max_value=1000,
        value=-1
    )

with col6:
    previous = st.number_input(
        "Previous Contacts",
        min_value=0,
        max_value=100,
        value=0
    )


col7, col8 = st.columns(2)

with col7:
    poutcome = st.selectbox(
        "Previous Campaign Outcome",
        [
            "unknown",
            "failure",
            "other",
            "success"
        ]
    )

with col8:
    st.markdown(
        """
        <div class="model-box">
            <div class="model-title">PREDICTION MODEL</div>
            <div class="model-text">
                Random Forest + LightGBM<br>
                Soft Voting Ensemble
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# BUTTON
# =========================================================

st.markdown("<br>", unsafe_allow_html=True)

predict_button = st.button(
    "PREDICT CUSTOMER SUBSCRIPTION"
)


# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    try:

        # ---------------------------------------------
        # CREATE INPUT DATAFRAME
        # ---------------------------------------------

        input_data = pd.DataFrame(
            {
                "age": [age],
                "job": [job],
                "marital": [marital],
                "education": [education],
                "default": [
                    1 if default == "yes" else 0
                ],
                "balance": [balance],
                "housing": [
                    1 if housing == "yes" else 0
                ],
                "loan": [
                    1 if loan == "yes" else 0
                ],
                "contact": [contact],
                "day": [day],
                "month": [month],
                "duration": [duration],
                "campaign": [campaign],
                "pdays": [pdays],
                "previous": [previous],
                "poutcome": [poutcome]
            }
        )


        # ---------------------------------------------
        # ONE-HOT ENCODING
        # ---------------------------------------------

        nominal_cols = [
            "job",
            "marital",
            "education",
            "contact",
            "month",
            "poutcome"
        ]

        input_data = pd.get_dummies(
            input_data,
            columns=nominal_cols,
            drop_first=True
        )


        # ---------------------------------------------
        # SCALING
        # ---------------------------------------------

        numeric_cols = [
            "age",
            "balance",
            "day",
            "campaign",
            "pdays",
            "previous"
        ]

        input_data[numeric_cols] = scaler.transform(
            input_data[numeric_cols]
        )


        # ---------------------------------------------
        # MATCH TRAINING FEATURES
        # ---------------------------------------------

        if hasattr(rf_model, "feature_names_in_"):

            input_data = input_data.reindex(
                columns=rf_model.feature_names_in_,
                fill_value=0
            )


        # ---------------------------------------------
        # RANDOM FOREST
        # ---------------------------------------------

        rf_probability = rf_model.predict_proba(
            input_data
        )[0][1]


        # ---------------------------------------------
        # LIGHTGBM
        # ---------------------------------------------

        lgbm_probability = lgbm_model.predict_proba(
            input_data
        )[0][1]


        # ---------------------------------------------
        # ENSEMBLE
        # ---------------------------------------------

        ensemble_probability = (
            rf_probability + lgbm_probability
        ) / 2


        # ---------------------------------------------
        # THRESHOLD
        # ---------------------------------------------

        threshold = 0.35

        prediction = int(
            ensemble_probability >= threshold
        )

        probability = ensemble_probability * 100


        # ---------------------------------------------
        # RESULT
        # ---------------------------------------------

        if prediction == 1:
            result_text = "Customer is likely to subscribe"
            icon = "✓"
        else:
            result_text = "Customer is unlikely to subscribe"
            icon = "—"


        st.markdown(
            """
            <div class="result-box">
                <div class="result-label">
                    AI PREDICTION
                </div>
            """,
            unsafe_allow_html=True
        )

        st.subheader(result_text)
        st.write(f"Subscription Probability: {probability:.2f}%") 


        # =================================================
        # METRICS
        # =================================================

        st.markdown("<br>", unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)

        with m1:
            st.metric(
                "Ensemble Probability",
                f"{probability:.2f}%"
            )

        with m2:
            st.metric(
                "Decision Threshold",
                "35%"
            )

        with m3:
            st.metric(
                "Prediction Model",
                "RF + LightGBM"
            )


        # =================================================
        # AI INSIGHTS
        # =================================================

        st.markdown(
            '<div class="section-title">AI Insights</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-subtitle">'
            'Quick interpretation of the prediction.'
            '</div>',
            unsafe_allow_html=True
        )


        i1, i2 = st.columns(2)


        with i1:

            if prediction == 1:
                explanation = (
                    "The customer's predicted probability "
                    "is above the optimized 35% decision threshold."
                )
            else:
                explanation = (
                    "The customer's predicted probability "
                    "is below the optimized 35% decision threshold."
                )

            st.subheader("Prediction Explanation")
            st.write(explanation)


        with i2:

            if prediction == 1:
                action = (
                    "Prioritize this customer for "
                    "marketing follow-up."
                )
            else:
                action = (
                    "Consider alternative targeting "
                    "strategies for this customer."
                )

            st.subheader("Recommended Action")

            if prediction == 1:
                st.markdown(
                    f"""
                    <p style="
                        background-color: #EEF1E4;
                        color: #3F4825;
                        padding: 15px 18px;
                        border-radius: 12px;
                        font-weight: 600;
                        font-size: 16px;
                    ">
                        {action}
                    </p>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <p style="
                        background-color: #F5EFE1;
                        color: #5C4B25;
                        padding: 15px 18px;
                        border-radius: 12px;
                        font-weight: 600;
                        font-size: 16px;
                    ">
                        {action}
                    </p>
                    """,
                    unsafe_allow_html=True
                )

    except Exception as e:

        st.error("Prediction could not be completed.")
        st.exception(e)