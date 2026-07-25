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

st.markdown(
    """
    <div class="instructions-card">
        Fill in the applicant's details in the three sections below, then press
        <b>Predict Default Risk</b> to see the estimated probability of default.
    </div>
    """,
    unsafe_allow_html=True,
)

model, preprocessor = load_artifacts()

left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.markdown('<div class="section-header">Personal</div>', unsafe_allow_html=True)
    age = st.number_input("Age", min_value=18, max_value=75, value=35, step=1)
    education = st.selectbox("Education", ["High School", "Bachelor's", "Master's", "PhD"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    has_dependents = st.selectbox("Has Dependents", ["Yes", "No"])

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Employment & Income</div>', unsafe_allow_html=True)
    employment_type = st.selectbox("Employment Type", ["Full-time", "Part-time", "Self-employed", "Unemployed"])
    months_employed = st.number_input(
        "Months Employed", min_value=0, max_value=480, value=60, step=1,
        help="How many months you've been at your current job.",
    )
    income = st.number_input(
        "Annual Income ($)", min_value=15000, max_value=200000, value=60000, step=1000,
    )
    credit_score = st.number_input(
        "Credit Score", min_value=300, max_value=850, value=650, step=1,
        help="Usually between 300 and 850 — check your bank app or credit report.",
    )

with right_col:
    st.markdown('<div class="section-header">Loan Details</div>', unsafe_allow_html=True)
    loan_amount = st.number_input(
        "Loan Amount ($)", min_value=1000, max_value=250000, value=50000, step=1000,
    )
    interest_rate = st.number_input(
        "Interest Rate (%)", min_value=1.0, max_value=30.0, value=10.0, step=0.1,
        help="The annual interest rate offered or expected on this loan.",
    )
    loan_term = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60])
    num_credit_lines = st.number_input(
        "Number of Open Credit Accounts", min_value=0, max_value=20, value=3, step=1,
        help="Total credit cards, loans, or credit lines you currently have open.",
    )
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
        "LoanTerm": loan_term,
        "Education": education, "EmploymentType": employment_type,
        "MaritalStatus": marital_status, "HasMortgage": has_mortgage,
        "HasDependents": has_dependents, "LoanPurpose": loan_purpose,
        "HasCoSigner": has_cosigner,
    }
    try:
        prediction, proba = predict_default(input_dict, model, preprocessor)
    except ValueError as exc:
        st.error(f"Prediction failed: {exc}")
        st.stop()

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