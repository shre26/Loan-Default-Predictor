import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
MODEL_PATH = os.path.join(BASE_DIR, "models", "final_model.pkl")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "models", "preprocessor.pkl")
BINARY_MAP = {"Yes": 1, "No": 0}


def load_artifacts():
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    return model, preprocessor


def predict_default(input_dict, model, preprocessor):
    df = pd.DataFrame([input_dict])
    for col in ["HasMortgage", "HasDependents", "HasCoSigner"]:
        df[col] = df[col].map(BINARY_MAP)
    X = preprocessor.transform(df)
    proba = model.predict_proba(X)[0][1]
    prediction = int(proba >= 0.5)
    return prediction, proba