import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn


# ===============================
# MLFLOW CONFIG 
# ===============================
MLFLOW_URI = os.environ.get("MLFLOW_URI", "https://dagshub.com/AnggaBagusS/UAS-DSP.mlflow")
REGISTERED_MODEL_NAME = os.environ.get("REGISTERED_MODEL_NAME", "logistic_regression_cancer")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "1")

MODEL_PATH = os.environ.get("MODEL_PATH", "model/logistic_regression_cancer.pkl")


# ===============================
# GET MODEL FROM MLFLOW
# ===============================
def get_model():
    mlflow.set_tracking_uri(MLFLOW_URI)

    model_uri = f"models:/{REGISTERED_MODEL_NAME}/{MODEL_VERSION}"
    model = mlflow.sklearn.load_model(model_uri)

    os.makedirs(os.path.dirname(MODEL_PATH) or "model", exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    return model


# ===============================
# LOAD MODEL LOCAL
# ===============================
def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except FileNotFoundError:
        print(f"Model tidak ditemukan di {MODEL_PATH}")
        return None


# ===============================
# FEATURE LIST (IDENTIK)
# ===============================
def set_features():
    return [
        "Age",
        "Coughing of Blood",
        "Dust Allergy",
        "Passive Smoker",
        "OccuPational Hazards",
        "Air Pollution",
        "chronic Lung Disease",
        "Shortness of Breath",
        "Dry Cough",
        "Snoring",
        "Swallowing Difficulty"
    ]


# ===============================
# PREDICTION HELPER
# ===============================
def predict(model, input_data: dict):
    df = pd.DataFrame([input_data])
    return model.predict(df)[0]
