from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
import logging


# Configure logging
logging.basicConfig(
    filename = "logs/prediction.log",
    level = logging.INFO,
    format = "%(asctime)s - %(message)s"
)


app = FastAPI(
    title="Fraud Detection API",
    description = "Detects posible fraud transaction" ,
    version = "1.0.0"
)

# Load model once
model = joblib.load("models/fraud_model.pkl")
pipeline = joblib.load("models/fraud_detection_pipeline.pkl")

# Input schema
class Transaction(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float
    Hour: float # Added missing 'Hour' column

@app.get("/")
def home():
    return {"message": "Fraud Detection API is running"}


@app.get("/health")
def health():
    try:
        steps = None
        
        if hasattr(pipeline, "steps"):
            steps = [step[0] for step in pipeline.steps]

        return {
            "status": "healthy",
            "model": type(pipeline).__name__,
            "pipeline_steps": steps
        }

    except Exception as e:
        return {
            "status": "error",
            "detail": str(e)
        }


@app.post("/predict")
def predict(transaction: Transaction):
    try:
        # Convert the Pydantic model to a DataFrame
        data = pd.DataFrame([transaction.model_dump()])

        # Use the pipeline to make predictions
        pred = pipeline.predict(data)[0]

        if hasattr(pipeline, "predict_proba"):
            prob = pipeline.predict_proba(data)[0][1]
        else:
            prob = None

        # Log input + output
        logging.info(f"input={transaction.dict()}, prep={pred}, prob={prob}")

        return {
            "fraud_prediction": int(pred),
            "fraud_probability": float(prob) if prob is not None else None
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))    

    data["prediction"] = pred
    data.to_csv("data/new_data.csv", mode='a', header=False, index=False)