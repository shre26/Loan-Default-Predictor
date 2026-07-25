import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
MODEL_PATH = os.path.join(BASE_DIR, "models", "final_model.pkl")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "models", "preprocessor.pkl")
BINARY_MAP = {"Yes": 1, "No": 0}

# The trained model only recognizes these categories. Indian-context labels
# shown in the UI are mapped onto them below — see README for the caveats
# this introduces (e.g. Gold loans have no true equivalent).
EMPLOYMENT_TYPE_MAP = {
    "Government Employee": "Full-time",
    "Private Employee": "Full-time",
    "Self Employed": "Self-employed",
    "Farmer": "Self-employed",
    "Housewife": "Unemployed",
    "Unemployed": "Unemployed",
}

LOAN_PURPOSE_MAP = {
    "Home": "Home",
    "Vehicle": "Auto",
    "Business": "Business",
    "Personal": "Other",
    "Gold": "Other",
}


def load_artifacts():
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    return model, preprocessor


def compute_net_income(gross_income, standard_deduction, lifestyle_expense):
    """
    Net income after standard deductions and lifestyle spending —
    this is what actually gets passed to the model as "Income",
    since the model's Income feature responds directly to this value.
    """
    return max(gross_income - standard_deduction - lifestyle_expense, 0)


def estimate_dti_ratio(loan_amount, interest_rate, loan_term, income):
    """
    Estimates debt-to-income ratio from the requested loan itself,
    since users don't reliably know this figure.
    Uses the standard monthly loan payment (EMI) formula, then divides
    by estimated monthly income. This is a simplification: it only
    accounts for the new loan, not any existing debt the applicant may have.
    """
    monthly_rate = (interest_rate / 100) / 12
    monthly_income = income / 12

    if monthly_rate == 0:
        monthly_payment = loan_amount / loan_term
    else:
        monthly_payment = (
            loan_amount * monthly_rate * (1 + monthly_rate) ** loan_term
            / ((1 + monthly_rate) ** loan_term - 1)
        )

    return monthly_payment / monthly_income


def predict_default(input_dict, model, preprocessor):
    input_dict = input_dict.copy()

    input_dict["EmploymentType"] = EMPLOYMENT_TYPE_MAP[input_dict["EmploymentType"]]
    input_dict["LoanPurpose"] = LOAN_PURPOSE_MAP[input_dict["LoanPurpose"]]

    input_dict["DTIRatio"] = estimate_dti_ratio(
        loan_amount=input_dict["LoanAmount"],
        interest_rate=input_dict["InterestRate"],
        loan_term=input_dict["LoanTerm"],
        income=input_dict["Income"],
    )

    df = pd.DataFrame([input_dict])
    for col in ["HasMortgage", "HasDependents", "HasCoSigner"]:
        df[col] = df[col].map(BINARY_MAP)

    X = preprocessor.transform(df)
    proba = model.predict_proba(X)[0][1]
    prediction = int(proba >= 0.5)
    return prediction, proba