import os
import sys

import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from predict import load_artifacts, predict_default

st.set_page_config(page_title="Loan Default Predictor", layout="wide")


def load_css(file_name):
    css_path = os.path.join(os.path.dirname(__file__), file_name)
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_css("style.css")

st.title("🏦 Loan Default Risk Assessment")
st.write("Predict the likelihood of loan default using applicant financial and demographic information.")

model, preprocessor = load_artifacts()

left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.markdown('<div class="section-header">Personal</div>', unsafe_allow_html=True)
    age = st.slider("Age", 18, 75, 35)
    education = st.selectbox("Education", ["High School", "Bachelor's", "Master's", "PhD"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    has_dependents = st.selectbox("Has Dependents", ["Yes", "No"])

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Employment & Income</div>', unsafe_allow_html=True)
    employment_type = st.selectbox("Employment Type", ["Full-time", "Part-time", "Self-employed", "Unemployed"])
    months_employed = st.slider("Months Employed", 0, 480, 60)
    income = st.slider("Annual Income ($)", 15000, 200000, 60000, step=1000)
    credit_score = st.slider("Credit Score", 300, 850, 650)

with right_col:
    st.markdown('<div class="section-header">Loan Details</div>', unsafe_allow_html=True)
    loan_amount = st.slider("Loan Amount ($)", 1000, 250000, 50000, step=1000)
    interest_rate = st.slider("Interest Rate (%)", 1.0, 30.0, 10.0, step=0.1)
    loan_term = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60])
    dti_ratio = st.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.3, step=0.01)
    num_credit_lines = st.slider("Number of Credit Lines", 0, 20, 3)
    has_mortgage = st.selectbox("Has Mortgage", ["Yes", "No"])
    loan_purpose = st.selectbox("Loan Purpose", ["Auto", "Business", "Education", "Home", "Other"])
    has_cosigner = st.selectbox("Has Co-Signer", ["Yes", "No"])

st.markdown("<br>", unsafe_allow_html=True)
predict_clicked = st.button("Predict Default Risk")

if predict_clicked:
    input_dict = {
        "Age": age, "Income": income, "LoanAmount": loan_amount,
        "CreditScore": credit_score, "MonthsEmployed": months_employed,
        "NumCreditLines": num_credit_lines, "InterestRate": interest_rate,
        "LoanTerm": loan_term, "DTIRatio": dti_ratio,
        "Education": education, "EmploymentType": employment_type,
        "MaritalStatus": marital_status, "HasMortgage": has_mortgage,
        "HasDependents": has_dependents, "LoanPurpose": loan_purpose,
        "HasCoSigner": has_cosigner,
    }
    prediction, proba = predict_default(input_dict, model, preprocessor)

    if prediction == 1:
        color = "#EF4444"
        status = "🔴 High Risk of Default"
    else:
        color = "#16A34A"
        status = "🟢 Low Risk of Default"

    st.markdown(
        f"""
        <div class="prediction-card">
            <div class="prediction-title">Default Probability</div>
            <div class="prediction-prob">{proba:.1%}</div>
            <hr style="border:1px solid #E5F2FF;">
            <div class="prediction-status" style="color:{color};">{status}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )