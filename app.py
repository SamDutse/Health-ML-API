from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load trained models
diabetes_model = joblib.load("diabetes_model.pkl")
cancer_model = joblib.load("cancer_model.pkl")

app = FastAPI(title="Health ML API")

# ========== INPUT SCHEMAS ==========

class DiabetesInput(BaseModel):
    features: list[float]  # 10 features

class CancerInput(BaseModel):
    features: list[float]  # 30 features


# ========== ENDPOINTS ==========

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict/diabetes")
def predict_diabetes(data: DiabetesInput):
    X = np.array(data.features).reshape(1, -1)
    prediction = diabetes_model.predict(X)[0]

    return {
        "model": "diabetes",
        "prediction": float(prediction),
        "unit": "disease progression score"
    }

@app.post("/predict/cancer")
def predict_cancer(data: CancerInput):
    X = np.array(data.features).reshape(1, -1)
    prediction = cancer_model.predict(X)[0]

    label = "malignant" if prediction == 0 else "benign"

    return {
        "model": "breast_cancer",
        "prediction": int(prediction),
        "class": label
    }
