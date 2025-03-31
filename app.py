
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load("hiv_model.pkl")

class PatientData(BaseModel):
    Procalcitonin: float
    Hemoglobin: float
    Lymphocytes: float
    Creatinine: float
    Platelets: float

@app.post("/predict")
def predict(data: PatientData):
    input_df = pd.DataFrame([data.dict().values()], columns=data.dict().keys())
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1] * 100
    return {
        "has_oi": bool(prediction),
        "probability": round(probability, 2)
    }
