
import streamlit as st
import pandas as pd
import numpy as np
import os
import uuid
from datetime import datetime
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import plotly.express as px

# ---------------- CONFIG ----------------
st.set_page_config(page_title="FinAI — Smart Loan Predictor", layout="wide")
DATA_DIR = "data"
APPLICANTS_CSV = os.path.join(DATA_DIR,"loan_applicants.csv")
os.makedirs(DATA_DIR, exist_ok=True)

# ---------------- CREATE CSV IF NOT EXIST ----------------
if not os.path.exists(APPLICANTS_CSV):
    cols = ["id","timestamp","name","email_or_phone","mode","gender","married","education",
            "employment_type","applicant_income","coapplicant_income","loan_amount","loan_term_months",
            "interest_rate","other_monthly_debt","property_value","credit_history",
            "score","confidence","decision","breakdown"]
    pd.DataFrame(columns=cols).to_csv(APPLICANTS_CSV,index=False)

# ---------------- HELPER FUNCTIONS ----------------
def emi_monthly(P, annual_rate, months):
    r = annual_rate/12/100
    if r==0: return P/months
    return (P*r*(1+r)**months)/((1+r)**months-1)

def compute_dti(emi, other_debt, income):
    return (emi + other_debt)/max(income,1)

def compute_ltv(loan_amount, property_value):
    return loan_amount/max(property_value,1)

def compute_score(features):
    # Banking formulas
    emi = emi_monthly(features["loan_amount"], features["interest_rate"], features["loan_term_months"])
    dti = compute_dti(emi, features.get("other_monthly_debt",0), features["applicant_income"]+features["coapplicant_income"])
    ltv = compute_ltv(features["loan_amount"], features.get("property_value",features["loan_amount"]*1.2))
    income_score = np.log1p(features["applicant_income"]+features["coapplicant_income"])/np.log1p(200000)*20
    credit_score = 30 if features.get("credit_history",0)==1 else 0
    emp_score = 3 if features.get("employment_type","Salaried").lower()=="salaried" else 0
    dti_score = 25 if dti<=0.35 else 10 if dti<=0.45 else 0
    ltv_score = 15 if ltv<=0.7 else 5 if ltv<=0.9 else 0
    raw = income_score+credit_score+emp_score+dti_score+ltv_score
    score = min(100,max(0,raw))
    confidence = min(0.98,0.55 + (score/100)*0.40)
    breakdown = {"emi":round(emi,2),"dti":round(dti,2),"ltv":round(ltv,2),
                 "income_score":round(income_score,2),"credit_score_component":credit_score,
                 "employment_component":emp_score,"dti_component":dti_score,"ltv_component":ltv_score,
                 "raw_sum":round(raw,2)}
    return round(score,2), round(confidence,2), breakdown

def save_application(rec):
    df = pd.read_csv(APPLICANTS_CSV)
    df = pd.concat([df,pd.DataFrame([rec])],ignore_index=True)
    df.to_csv(APPLICANTS_CSV,index=False)

def train_model(df):
    X = df[["applicant_income","coapplicant_income","loan_amount","loan_term_months","interest_rate","other_monthly_debt","property_value","credit_history"]]
    y = (df["score"]>=55).astype(int)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    svm = SVC(probability=True)
    lr = LogisticRegression()
    # Hybrid: average probability
    svm.fit(X_scaled,y)
    lr.fit(X_scaled,y)
    return {"svm":svm,"lr":lr,"scaler":scaler}

def predict_loan(model, features):
    X = np.array([[features["applicant_income"],features["coapplicant_income"],features["loan_amount"],
                   features["loan_term_months"],features["interest_rate"],features.get("other_monthly_debt",0),
                   features.get("property_value",features["loan_amount"]*1.2),features.get("credit_history",1)]])
    X_scaled = model["scaler"].transform(X)
    prob_svm = model["svm"].predict_proba(X_scaled)[0][1]
    prob_lr = model["lr"].predict_proba(X_scaled)[0][1]
    prob = (prob_svm+prob_lr)/2
    decision = "APPROVED" if prob>=0.55 else "REJECTED"
    return round(prob*100,2), decision

# ---------------- SIDEBAR LOGIN ----------------
st.sidebar.header("Login Portal")
mode = st.sidebar.selectbox("Mode",["Applicant","Bank Admin"])
if mode=="Applicant":
    login_mode = st.sidebar.radio("Login via",["Email","Phone"])
    email_or_phone = st.sidebar.text_input(f"{login_mode}")
    otp = st.sidebar.text_input("Enter OTP (mock, 6 digits)")
    if st.sidebar.button("Verify"):
        if email_or_phone.strip()!="" and otp.strip()!="":
            st.session_state["app_authenticated"]=True
            st.session_state["app_user"]=email_or_phone
            st.sidebar.success("Login successful")
        else:
            st.sidebar.error("Enter login and OTP")
elif mode=="Bank Admin":
    admin_pass = st.sidebar.text_input("Admin Password", type="password")
    if st.sidebar.button("Login"):
        if admin_pass=="FinAIAdmin123":
            st.session_state["admin_authenticated"]=True
            st.sidebar.success("Admin login successful")
        else:
            st.sidebar.error("Wrong password")

# ---------------- MAIN ----------------
st.title("FinAI — Smart Loan Predictor")

if mode=="Applicant" and st.session_state.get("app_authenticated",False):
    st.header("Loan Application Form")
    with st.form("loan_form"):
        cols = st.columns(3)
        name = cols[0].text_input("Full Name")
        gender = cols[0].selectbox("Gender",["Male","Female","Other"])
        married = cols[0].selectbox("Married",["Yes","No"])
        education = cols[0].selectbox("Education",["Graduate","Not Graduate"])
        employment_type = cols[1].selectbox("Employment Type",["Salaried","Self-Employed","Unemployed"])
        applicant_income = cols[1].number_input("Applicant Monthly Income (₹)",min_value=0.0,value=25000.0)
        coapplicant_income = cols[1].number_input("Co-Applicant Monthly Income (₹)",min_value=0.0,value=0.0)
        loan_amount = cols[2].number_input("Loan Amount (₹)",min_value=10000.0,value=500000.0)
        loan_term_years = cols[2].selectbox("Loan Term (Years)",[5,10,15,20,25,30])
        loan_term_months = int(loan_term_years*12)
        interest_rate = cols[2].number_input("Interest Rate (%)",min_value=0.0,value=7.5)
        other_debt = cols[2].number_input("Other Monthly Liabilities (₹)",min_value=0.0,value=0.0)
        property_value = st.number_input("Property Value (₹)",min_value=0.0,value=700000.0)
        credit_history = st.selectbox("Credit History (1=Good,0=Poor)",[1,0])
        submitted = st.form_submit_button("Submit Application")

    if submitted:
        features = {"applicant_income":float(applicant_income),"coapplicant_income":float(coapplicant_income),
                    "loan_amount":float(loan_amount),"loan_term_months":loan_term_months,
                    "interest_rate":float(interest_rate),"other_monthly_debt":float(other_debt),
                    "property_value":float(property_value),"credit_history":int(credit_history),
                    "employment_type":employment_type}
        score, confidence, breakdown = compute_score(features)
        decision = "APPROVED" if score>=55 else "REJECTED"
        st.subheader(f"Decision: {decision}  —  Score: {score}/100")
        st.metric("Confidence",f"{int(confidence*100)}%")
        st.json(breakdown)
        rec = {"id":str(uuid.uuid4()),"timestamp":datetime.utcnow().isoformat(),"name":name,
               "email_or_phone":st.session_state.get("app_user","unknown"),"mode":login_mode,
               "gender":gender,"married":married,"education":education,"employment_type":employment_type,
               "applicant_income":applicant_income,"coapplicant_income":coapplicant_income,
               "loan_amount":loan_amount,"loan_term_months":loan_term_months,"interest_rate":interest_rate,
               "other_monthly_debt":other_debt,"property_value":property_value,"credit_history":credit_history,
               "score":score,"confidence":confidence,"decision":decision,"breakdown":str(breakdown)}
        save_application(rec)
        st.success("Application saved successfully!")

elif mode=="Bank Admin" and st.session_state.get("admin_authenticated",False):
    st.header("Bank Admin Dashboard")
    df = pd.read_csv(APPLICANTS_CSV)
    if df.empty:
        st.info("No applications yet")
    else:
        total=len(df); approved=len(df[df["decision"]=="APPROVED"]); rejected=total-approved
        c1,c2,c3=st.columns(3); c1.metric("Total Applications",total); c2.metric("Approved",approved); c3.metric("Rejected",rejected)
        st.subheader("Applications Table")
        st.dataframe(df[["timestamp","name","email_or_phone","applicant_income","coapplicant_income","loan_amount","decision"]])
        st.subheader("Plots")
        fig = px.scatter(df, x="applicant_income", y="loan_amount", color="decision", hover_data=["name"])
        st.plotly_chart(fig, use_container_width=True)
        fig2 = px.histogram(df, x="score", nbins=20, color="decision")
        st.plotly_chart(fig2, use_container_width=True)

else:
    st.info("Login to continue")
