
import streamlit as st
import pandas as pd
import numpy as np
import os
import uuid
from datetime import datetime
import plotly.express as px

st.set_page_config(page_title="Professional Loan Eligibility System", page_icon="🏦", layout="wide")

DATA_DIR = "data"
DATA_FILE = os.path.join(DATA_DIR, "loan_enquiries.csv")
OTP_FILE = os.path.join(DATA_DIR, "otps.csv")
ADMIN_EMAILS = ["bankadmin@bankdemo.com"]  # Add admin emails here

os.makedirs(DATA_DIR, exist_ok=True)

# Initialize files if missing
if not os.path.exists(DATA_FILE):
    cols = [
        "id", "timestamp", "user_email", "gender", "married", "education",
        "self_employed", "applicant_income", "coapplicant_income", "loan_amount",
        "loan_term", "interest_rate", "credit_history", "property_area", "property_value",
        "loan_status", "score", "confidence", "explanation"
    ]
    pd.DataFrame(columns=cols).to_csv(DATA_FILE, index=False)

if not os.path.exists(OTP_FILE):
    pd.DataFrame(columns=["email", "otp", "created_at"]).to_csv(OTP_FILE, index=False)

# ---------------------------
# Utilities
# ---------------------------
def generate_otp():
    return str(np.random.randint(100000, 999999))

def send_otp_simulated(email):
    otp = generate_otp()
    rec = {"email": email, "otp": otp, "created_at": datetime.utcnow().isoformat()}
    df = pd.read_csv(OTP_FILE)
    df = pd.concat([df, pd.DataFrame([rec])], ignore_index=True)
    df.to_csv(OTP_FILE, index=False)
    return otp

def verify_otp(email, otp_input):
    df = pd.read_csv(OTP_FILE)
    rows = df[(df["email"] == email) & (df["otp"] == otp_input)]
    return len(rows) > 0

def append_enquiry(record: dict):
    df = pd.read_csv(DATA_FILE)
    df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    df.to_csv(DATA_FILE, index=False)

def load_enquiries():
    return pd.read_csv(DATA_FILE)

def human_time(ts):
    try:
        return datetime.fromisoformat(ts).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ts

# ---------------------------
# Authentication UI
# ---------------------------
def auth_ui():
    st.sidebar.header("Account")
    if "auth_email" not in st.session_state:
        st.session_state.auth_email = ""
    if "auth_role" not in st.session_state:
        st.session_state.auth_role = None
    if "auth_ok" not in st.session_state:
        st.session_state.auth_ok = False

    if not st.session_state.auth_ok:
        mode = st.sidebar.selectbox("I am a", ["Applicant", "Bank Admin"])
        email = st.sidebar.text_input("Email", value=st.session_state.auth_email, placeholder="you@example.com")
        st.session_state.auth_email = email
        if st.sidebar.button("Send OTP"):
            if email.strip() == "":
                st.sidebar.error("Enter a valid email.")
            else:
                otp = send_otp_simulated(email)
                st.sidebar.success(f"OTP sent to {email} (simulated).")
                if st.sidebar.checkbox("Show OTP (demo only)"):
                    st.sidebar.info(f"OTP: {otp}")
        otp_input = st.sidebar.text_input("Enter OTP (6 digits)")
        if st.sidebar.button("Verify OTP"):
            if verify_otp(email, otp_input):
                st.session_state.auth_ok = True
                st.session_state.auth_role = "admin" if email in ADMIN_EMAILS else "applicant"
                st.sidebar.success("Logged in successfully.")
                st.session_state.last_login = datetime.utcnow().isoformat()
            else:
                st.sidebar.error("Invalid OTP. Try again.")
    else:
        role = st.session_state.auth_role
        st.sidebar.success(f"Logged in as: {st.session_state.auth_email} ({role})")
        if st.sidebar.button("Logout"):
            st.session_state.auth_ok = False
            st.session_state.auth_email = ""
            st.session_state.auth_role = None
            st.experimental_rerun()

# ---------------------------
# Banking Formula-based Loan Scoring
# ---------------------------
def compute_loan_score(features: dict):
    """
    Compute realistic loan score using banking formulas:
    DTI, LTV, Income weighting, Credit history, Term, Property area.
    """
    score = 0.0
    explanation = {}

    # Credit history weight
    if features["credit_history"] == 1:
        score += 30
        explanation["credit_history"] = "+30 (good credit)"
    else:
        explanation["credit_history"] = "+0 (poor credit)"

    # Debt-to-Income Ratio (DTI)
    monthly_rate = features["interest_rate"]/12/100
    n = features["loan_term"]
    emi = (features["loan_amount"] * 1000 * monthly_rate * (1 + monthly_rate)**n) / ((1 + monthly_rate)**n - 1)
    total_income = features["applicant_income"] + features["coapplicant_income"]
    dti = emi / total_income
    if dti <= 0.4:
        score += 25
        explanation["DTI"] = f"+25 (DTI={dti:.2f} ≤ 0.4)"
    elif dti <= 0.6:
        score += 10
        explanation["DTI"] = f"+10 (DTI={dti:.2f})"
    else:
        explanation["DTI"] = f"+0 (DTI={dti:.2f} high)"

    # Loan-to-Value Ratio (LTV)
    ltv = features["loan_amount"]*1000 / features["property_value"]
    if ltv <= 0.8:
        score += 15
        explanation["LTV"] = f"+15 (LTV={ltv:.2f} ≤ 0.8)"
    elif ltv <= 0.9:
        score += 5
        explanation["LTV"] = f"+5 (LTV={ltv:.2f})"
    else:
        explanation["LTV"] = f"+0 (LTV={ltv:.2f} high)"

    # Income weighting
    income_score = np.log(features["applicant_income"]+1) + 0.5*np.log(features["coapplicant_income"]+1)
    score += income_score
    explanation["IncomeScore"] = f"+{income_score:.2f} (log scaled)"

    # Term adjustment
    term_adj = 0.02 * (n/12)
    score += term_adj
    explanation["TermAdj"] = f"+{term_adj:.2f} (longer term)"

    # Property area adjustment
    if features["property_area"] == "Urban":
        score += 3
        explanation["PropertyArea"] = "+3 (urban)"
    elif features["property_area"] == "Semiurban":
        score += 1.5
        explanation["PropertyArea"] = "+1.5 (semiurban)"
    else:
        explanation["PropertyArea"] = "+0 (rural)"

    # Self-employed & education minor adjustment
    if features["self_employed"] == "No":
        score += 3
        explanation["Employment"] = "+3 (salaried)"
    else:
        explanation["Employment"] = "+0 (self-employed)"
    if features["education"] == "Graduate":
        score += 2
        explanation["Education"] = "+2 (graduate)"
    else:
        explanation["Education"] = "+0 (not graduate)"

    # Normalize score 0-100
    raw_score = min(95, score)
    confidence = 0.6 + (raw_score/100)*0.35
    confidence = min(0.99, confidence)

    return int(raw_score), float(confidence), explanation

# ---------------------------
# Main UI
# ---------------------------
def main():
    st.markdown("<h2 style='color:#003366'>Professional Loan Eligibility System</h2>", unsafe_allow_html=True)

    auth_ui()
    if "auth_ok" not in st.session_state or not st.session_state.auth_ok:
        st.info("Login from sidebar to continue. Use OTP reveal for demo.")
        return

    user_email = st.session_state.auth_email
    role = st.session_state.auth_role

    if role == "applicant":
        st.header("Apply for Loan")
        with st.form("loan_form"):
            gender = st.selectbox("Gender", ["Male", "Female"])
            married = st.selectbox("Married", ["Yes", "No"])
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
            self_employed = st.selectbox("Self Employed", ["No", "Yes"])
            applicant_income = st.number_input("Applicant Income (₹)", value=20000)
            coapplicant_income = st.number_input("Co-applicant Income (₹)", value=0)
            loan_amount = st.number_input("Loan Amount (₹)", value=500000)
            loan_term = st.selectbox("Loan Term (months)", [120,180,240,300,360])
            interest_rate = st.number_input("Interest Rate (%)", value=7.5)
            credit_history = st.selectbox("Credit History (1=good,0=poor)", [1,0])
            property_area = st.selectbox("Property Area", ["Urban","Semiurban","Rural"])
            property_value = st.number_input("Estimated Property Value (₹)", value=700000)
            submitted = st.form_submit_button("Submit")

        if submitted:
            features = {
                "applicant_income": applicant_income,
                "coapplicant_income": coapplicant_income,
                "loan_amount": loan_amount/1000,  # thousands
                "loan_term": loan_term,
                "interest_rate": interest_rate,
                "credit_history": credit_history,
                "property_area": property_area,
                "property_value": property_value,
                "self_employed": self_employed,
                "education": education
            }
            score, confidence, explanation = compute_loan_score(features)
            approved = score >= 50
            status = "Y" if approved else "N"
            ts = datetime.utcnow().isoformat()
            rec = {
                "id": str(uuid.uuid4()),
                "timestamp": ts,
                "user_email": user_email,
                "gender": gender,
                "married": married,
                "education": education,
                "self_employed": self_employed,
                "applicant_income": applicant_income,
                "coapplicant_income": coapplicant_income,
                "loan_amount": loan_amount/1000,
                "loan_term": loan_term,
                "interest_rate": interest_rate,
                "credit_history": credit_history,
                "property_area": property_area,
                "property_value": property_value,
                "loan_status": status,
                "score": score,
                "confidence": round(confidence,4),
                "explanation": "; ".join([f"{k}:{v}" for k,v in explanation.items()])
            }
            append_enquiry(rec)

            st.markdown(f"### Result: {'APPROVED ✅' if approved else 'REJECTED ❌'}")
            st.write(f"Score: {score}/100 | Confidence: {confidence*100:.1f}%")
            st.write("Explanation of contributing factors:")
            st.table(pd.DataFrame(list(explanation.items()), columns=["Factor","Contribution"]))

    elif role == "admin":
        st.header("Bank / Admin Dashboard")
        df = load_enquiries()
        if df.empty:
            st.info("No enquiries yet.")
            return
        df["timestamp_readable"] = df["timestamp"].apply(human_time)
        st.dataframe(df[[
            "timestamp_readable", "user_email","applicant_income","coapplicant_income",
            "loan_amount","loan_term","credit_history","property_area","loan_status","score","confidence"
        ]])

        st.markdown("### Loan Approval Distribution")
        fig = px.pie(df, names="loan_status", title="Approved vs Rejected")
        st.plotly_chart(fig)

        st.markdown("### Loan Score Histogram")
        fig2 = px.histogram(df, x="score", nbins=20, title="Score Distribution")
        st.plotly_chart(fig2)

        csv_exp = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Full Enquiries CSV", data=csv_exp, file_name="loan_enquiries.csv")

# ---------------------------
# Run App
# ---------------------------
if __name__ == "__main__":
    main()
