from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import logging

# ========== LOGGING CONFIG ==========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# ========== LOAD MODELS ==========
diabetes_model = joblib.load("diabetes_model.pkl")
cancer_model = joblib.load("cancer_model.pkl")

app = FastAPI(title="Health ML API")

# ========== INPUT SCHEMAS ==========
class DiabetesInput(BaseModel):
    features: list[float]

class CancerInput(BaseModel):
    features: list[float]

# ========== ENDPOINTS ==========

@app.get("/health")
def health_check():
    logger.info("Health check called")
    return {"status": "ok"}

@app.post("/predict/diabetes")
def predict_diabetes(data: DiabetesInput):
    logger.info(f"Diabetes request: {data.features}")

    X = np.array(data.features).reshape(1, -1)
    prediction = diabetes_model.predict(X)[0]

    logger.info(f"Diabetes prediction: {prediction}")

    return {
        "model": "diabetes",
        "prediction": float(prediction),
        "unit": "disease progression score"
    }

@app.post("/predict/cancer")
def predict_cancer(data: CancerInput):
    logger.info(f"Cancer request: {data.features}")

    X = np.array(data.features).reshape(1, -1)
    prediction = cancer_model.predict(X)[0]

    label = "malignant" if prediction == 0 else "benign"

    logger.info(f"Cancer prediction: {label}")

    return {
        "model": "breast_cancer",
        "prediction": int(prediction),
        "class": label
    }
