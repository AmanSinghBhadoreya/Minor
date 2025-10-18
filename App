import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import os

st.set_page_config(page_title="Loan Eligibility Predictor", page_icon="🏦", layout="centered")

# Load model & scaler
model = joblib.load("model/loan_model.pkl")
scaler = joblib.load("model/scaler.pkl")

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #f4f8fb; }
    .stButton>button {
        background-color: #004aad;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-weight: bold;
    }
    .stButton>button:hover { background-color: #00367a; }
    </style>
""", unsafe_allow_html=True)

st.title("🏦 Loan Eligibility Prediction System")
st.subheader("An ML-based Financial Screening Tool using SVM")

st.markdown("---")

# Input form
col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
with col2:
    applicant_income = st.number_input("Applicant Income", min_value=0)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
    loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0)
    loan_term = st.selectbox("Loan Amount Term", [180, 240, 300, 360])
    credit_history = st.selectbox("Credit History", [1, 0])

# Prediction
if st.button("🔍 Predict Eligibility"):
    # Prepare input
    input_data = pd.DataFrame({
        'Gender': [1 if gender == 'Male' else 0],
        'Married': [1 if married == 'Yes' else 0],
        'Education': [1 if education == 'Graduate' else 0],
        'Self_Employed': [1 if self_employed == 'Yes' else 0],
        'ApplicantIncome': [applicant_income],
        'CoapplicantIncome': [coapplicant_income],
        'LoanAmount': [loan_amount],
        'Loan_Amount_Term': [loan_term],
        'Credit_History': [credit_history],
        'Property_Area': [0 if property_area == 'Rural' else (1 if property_area == 'Semiurban' else 2)]
    })

    scaled = scaler.transform(input_data)
    pred = model.predict(scaled)[0]
    proba = model.predict_proba(scaled)[0][pred]

    result = "✅ Loan Approved" if pred == 1 else "❌ Loan Rejected"
    st.success(f"**Prediction:** {result}")
    st.write(f"**Confidence:** {proba*100:.2f}%")

    # Log prediction
    os.makedirs("logs", exist_ok=True)
    log_data = input_data.copy()
    log_data["Prediction"] = result
    log_data["Confidence"] = round(proba*100, 2)
    log_data.to_csv("logs/prediction_history.csv", mode='a', header=not os.path.exists("logs/prediction_history.csv"), index=False)

    # Explainability (SHAP)
    st.markdown("### 🧠 Model Explainability (SHAP)")
    explainer = shap.Explainer(model)
    shap_values = explainer(scaled)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    shap.summary_plot(shap_values, input_data, show=False)
    st.pyplot(bbox_inches='tight')
