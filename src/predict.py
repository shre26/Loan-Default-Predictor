import os

import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "final_model.pkl")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "models", "preprocessor.pkl")
BINARY_MAP = {"Yes": 1, "No": 0}


def load_artifacts():
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    return model, preprocessor


def estimate_dti_ratio(income, loan_amount, interest_rate, loan_term):
    """
    Most applicants don't know their debt-to-income ratio offhand, so it's
    estimated here instead of asked for directly. Uses the standard loan
    amortization formula to work out the monthly payment, then compares
    that to estimated monthly income.
    """
    monthly_income = income / 12
    monthly_rate = interest_rate / 100 / 12

    if monthly_rate == 0:
        monthly_payment = loan_amount / loan_term
    else:
        monthly_payment = (
            loan_amount * monthly_rate * (1 + monthly_rate) ** loan_term
            / ((1 + monthly_rate) ** loan_term - 1)
        )

    dti_ratio = monthly_payment / monthly_income
    return float(np.clip(dti_ratio, 0.0, 1.0))


def predict_default(input_dict, model, preprocessor):
    input_dict = input_dict.copy()
    input_dict["DTIRatio"] = estimate_dti_ratio(
        income=input_dict["Income"],
        loan_amount=input_dict["LoanAmount"],
        interest_rate=input_dict["InterestRate"],
        loan_term=input_dict["LoanTerm"],
    )

    df = pd.DataFrame([input_dict])
    for col in ["HasMortgage", "HasDependents", "HasCoSigner"]:
        df[col] = df[col].map(BINARY_MAP)

    X = preprocessor.transform(df)
    proba = model.predict_proba(X)[0][1]
    prediction = int(proba >= 0.5)
    return prediction, proba