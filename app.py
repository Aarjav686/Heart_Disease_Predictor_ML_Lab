import streamlit as st
import pandas as pd
import joblib
import json
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Heart Disease Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    /* Main container background */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
        font-family: 'Inter', sans-serif;
    }
    
    /* Clean up the sidebar */
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    
    /* Title typography */
    h1 {
        color: #58a6ff;
        font-weight: 800;
        font-size: 2.8rem;
        margin-bottom: 0.5rem;
    }
    
    h2, h3 {
        color: #8b949e;
        font-weight: 600;
    }

    /* Cards */
    .result-card {
        padding: 2rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px -10px rgba(0,0,0,0.5);
        margin: 2rem 0;
        animation: fadeIn 0.8s ease;
    }
    
    .card-safe {
        background: linear-gradient(135deg, #2ea043 0%, #238636 100%);
    }
    
    .card-danger {
        background: linear-gradient(135deg, #f85149 0%, #da3633 100%);
    }

    .probability-text {
        font-size: 4rem;
        font-weight: 800;
        margin: 0;
        line-height: 1;
    }
    
    .status-text {
        font-size: 1.5rem;
        font-weight: 600;
        margin-top: 1rem;
        opacity: 0.9;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

# --- LOAD ASSETS ---
@st.cache_resource
def load_models():
    if not (os.path.exists("logistic_model.joblib") and os.path.exists("scaler.joblib")):
        st.error("Model files not found! Please run train_and_save.py first.")
        return None, None, None
    
    model = joblib.load("logistic_model.joblib")
    scaler = joblib.load("scaler.joblib")
    with open("expected_columns.json", "r") as f:
        expected_cols = json.load(f)
    return model, scaler, expected_cols

model, scaler, expected_cols = load_models()

# --- UI HEADER ---
st.markdown("<h1>Heart Disease Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h3>Empowering health decisions with machine learning.</h3>", unsafe_allow_html=True)
st.markdown("---")

# --- SIDEBAR INPUTS ---
st.sidebar.markdown("## 🩺 Patient Inputs")
st.sidebar.markdown("Please enter the diagnostic features below.")

col1, col2, col3 = st.columns([1, 1, 1])

# Form to collect inputs
with st.sidebar.form("predict_form"):
    
    age = st.slider("Age", 20, 100, 55)
    sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
    
    chest_pain = st.selectbox("Chest Pain Type", options=[1, 2, 3, 4], format_func=lambda x: {
        1: "Typical Angina", 2: "Atypical Angina", 3: "Non-anginal", 4: "Asymptomatic"
    }[x])
    
    resting_bp = st.slider("Resting Blood Pressure", 90, 200, 130)
    cholesterol = st.slider("Cholesterol (mg/dl)", 100, 600, 250)
    
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    
    rest_ecg = st.selectbox("Resting ECG", options=[0, 1, 2], format_func=lambda x: {
        0: "Normal", 1: "ST-T wave abnormality", 2: "Left ventricular hypertrophy"
    }[x])
    
    max_hr = st.slider("Maximum Heart Rate", 60, 220, 150)
    exercise_angina = st.selectbox("Exercise Induced Angina", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    
    oldpeak = st.slider("ST depression induced by exercise", 0.0, 6.2, 1.0, 0.1)
    
    slope = st.selectbox("Slope of peak exercise ST segment", options=[1, 2, 3], format_func=lambda x: {
        1: "Upsloping", 2: "Flat", 3: "Downsloping"
    }[x])
    
    num_vessels = st.slider("Number of major vessels (0-3)", 0, 3, 0)
    
    thal = st.selectbox("Thalassemia", options=[3, 6, 7], format_func=lambda x: {
        3: "Normal", 6: "Fixed Defect", 7: "Reversible Defect"
    }[x])

    submitted = st.form_submit_button("Run Analysis 🚀")

# --- PREDICTION LOGIC ---
if model and scaler and expected_cols:
    if submitted:
        # Build dictionary of raw feature inputs
        input_dict = {
            "age": age,
            "sex": sex,
            "chest_pain": chest_pain,
            "resting_bp": resting_bp,
            "cholesterol": cholesterol,
            "fasting_bs": fasting_bs,
            "rest_ecg": rest_ecg,
            "max_hr": max_hr,
            "exercise_angina": exercise_angina,
            "oldpeak": oldpeak,
            "slope": slope,
            "num_vessels": num_vessels,
            "thal": thal
        }
        
        # Convert to DataFrame
        df_input = pd.DataFrame([input_dict])
        
        # Apply get_dummies matching training
        df_input = pd.get_dummies(df_input, columns=[
            'chest_pain', 'rest_ecg', 'slope', 'thal'
        ])
        
        # Realign to expected columns (filling missing one-hot variables with False/0)
        df_input = df_input.reindex(columns=expected_cols, fill_value=0)
        
        # Convert True/False to 1/0 explicitly
        df_input = df_input.astype(float)
        
        # Scale the data
        X_scaled = scaler.transform(df_input)
        
        # Predict
        prediction = model.predict(X_scaled)[0]
        prediction_proba = model.predict_proba(X_scaled)[0]
        
        disease_prob = prediction_proba[1] * 100
        safe_prob = prediction_proba[0] * 100
        
        with col2:
            if prediction == 1:
                st.markdown(f"""
                <div class="result-card card-danger">
                    <p class="probability-text">{disease_prob:.1f}%</p>
                    <p class="status-text">High Risk of Heart Disease Detected</p>
                    <p style="opacity: 0.8; margin-top: 10px; font-size: 0.9rem;">
                        Our model recognizes patterns correlating with positive diagnosis. Please consult a medical professional immediately.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card card-safe">
                    <p class="probability-text">{safe_prob:.1f}%</p>
                    <p class="status-text">Low Risk Profile Pattern</p>
                    <p style="opacity: 0.8; margin-top: 10px; font-size: 0.9rem;">
                        Our model predicts absence of heart disease traits based on your metrics. Keep up the healthy lifestyle!
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
    else:
        with col2:
            st.info("👈 Enter patient vitals in the sidebar and click **Run Analysis** to see the prediction results.")
