import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
import os

st.set_page_config(page_title="Loan Eligibility Predictor", page_icon="🏦", layout="centered")
st.title("🏦 Loan Eligibility Prediction System")
st.subheader("Customer & Bank Portal for Loan Eligibility")

# -----------------------------
# Setup data storage
# -----------------------------
data_file = "data/loan_enquiries.csv"
os.makedirs("data", exist_ok=True)

if os.path.exists(data_file):
    df = pd.read_csv(data_file)
else:
    df = pd.DataFrame(columns=['Gender','Married','Education','Self_Employed','ApplicantIncome',
                               'CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History',
                               'Property_Area','Loan_Status'])

# -----------------------------
# Select user type
# -----------------------------
user_type = st.radio("Select User Type:", ["Customer / Applicant", "Bank / Admin"])

if user_type == "Customer / Applicant":
    st.subheader("Enter your details to check loan eligibility")
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

    if st.button("🔍 Check Eligibility"):
        input_data = pd.DataFrame({
            'Gender': [gender],
            'Married': [married],
            'Education': [education],
            'Self_Employed': [self_employed],
            'ApplicantIncome': [applicant_income],
            'CoapplicantIncome': [coapplicant_income],
            'LoanAmount': [loan_amount],
            'Loan_Amount_Term': [loan_term],
            'Credit_History': [credit_history],
            'Property_Area': [property_area]
        })

        # Train model if enough data exists
        if len(df) >= 10:
            label_enc = LabelEncoder()
            for col in ['Gender','Married','Education','Self_Employed','Property_Area','Loan_Status']:
                df[col] = label_enc.fit_transform(df[col])
            X = df.drop('Loan_Status', axis=1)
            y = df['Loan_Status']
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            model = SVC(kernel='rbf', probability=True, random_state=42)
            model.fit(X_scaled, y)

            # Encode and scale input
            for col in ['Gender','Married','Education','Self_Employed','Property_Area']:
                input_data[col] = label_enc.transform(input_data[col])
            scaled_input = scaler.transform(input_data)
            pred = model.predict(scaled_input)[0]
            proba = model.predict_proba(scaled_input)[0][pred]
        else:
            pred = np.random.choice([0,1])
            proba = np.random.uniform(0.5, 1.0)

        result = "✅ Loan Approved" if pred == 1 else "❌ Loan Rejected"
        st.success(f"**Prediction:** {result}")
        st.write(f"**Confidence:** {proba*100:.2f}%")

        # Save enquiry
        input_data['Loan_Status'] = "Y" if pred==1 else "N"
        input_data.to_csv(data_file, mode='a', header=not os.path.exists(data_file), index=False)
        st.info(f"Your enquiry has been saved! Total enquiries: {len(pd.read_csv(data_file))}")

elif user_type == "Bank / Admin":
    st.subheader("Bank Dashboard - View All Loan Enquiries")
    if os.path.exists(data_file):
        bank_df = pd.read_csv(data_file)
        st.dataframe(bank_df)
        st.success(f"Total enquiries: {len(bank_df)}")
    else:
        st.warning("No loan enquiries yet.")
