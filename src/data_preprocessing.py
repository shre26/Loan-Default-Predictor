import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

RAW_DATA_PATH = "../data/raw/loan_default.csv"

NUMERIC_FEATURES = [
    "Age", "Income", "LoanAmount", "CreditScore", "MonthsEmployed", "NumCreditLines", "InterestRate", "LoanTerm", "DTIRatio",
]
BINARY_FEATURES = ["HasMortgage", "HasDependents", "HasCoSigner"]
NOMINAL_FEATURES = ["Education", "EmploymentType",
                    "MaritalStatus", "LoanPurpose"]
TARGET = "Default"
DROP_COLS = ["LoanID"]


def load_data(path=RAW_DATA_PATH):
    df = pd.read_csv(path)
    return df.drop(columns=DROP_COLS)


def get_train_test_split(df, test_size=0.2, random_state=42):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    # stratify=y bcoz Default is imbalanced - keeps same proportion of 0,1 values in both train n test data
    return train_test_split(X, y, test_size=test_size,
                            random_state=random_state, stratify=y)


def encode_binary_features(X_train, X_test):
    X_train, X_test = X_train.copy(), X_test.copy()
    mapping = {"Yes": 1, "No": 0}
    for col in BINARY_FEATURES:
        X_train[col] = X_train[col].map(mapping)
        X_test[col] = X_test[col].map(mapping)
    return X_train, X_test


def build_preprocessor():
    """
    Scales numeric features, one-hot encodes nominal categoricals.
    Binary Yes/No columns are already 0/1 by this point, so they pass through.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("nom", OneHotEncoder(drop="first",
             handle_unknown="ignore"), NOMINAL_FEATURES)
        ],
        remainder="passthrough",
    )
