# 🚀 Credit Card Fraud Detection System (End-to-End ML + Deployment)

An end-to-end machine learning system to detect fraudulent credit card transactions, designed with a production-oriented workflow including model training, API deployment, Dockerization, logging, and retraining.

This project addresses the real-world challenge of fraud detection, where fraudulent transactions are extremely rare (highly imbalanced data) but critically important to identify. The focus is on maximizing recall (catching fraud cases) while maintaining a balanced F1-score for overall performance.

The system is built using Python with Scikit-learn and XGBoost for machine learning, a FastAPI-based REST API for serving predictions, and Docker for containerization. Models are serialized using Joblib and managed through manual versioning (e.g., `fraud_model_v1.pkl`, `fraud_model_v2.pkl`) instead of a centralized model registry.

---

## 🏗️ System Architecture

Data flows through the system as follows:

Data → Training → Model (.pkl) → API → Docker → User
↓
Logging Layer
↓
Retraining Pipeline

---

## 📁 Project Structure

```
fraud-detection-mlops/
│
├── app/
│   └── main.py              # FastAPI app
│
├── pipeline/
│   ├── train.py            # Model training
│   └── retrain.py          # Retraining pipeline
│
├── models/
│   ├── fraud_model_v1.pkl
│   └── fraud_model_v2.pkl
│
├── data/
│   ├── processed_data.csv
│   └── new_data.csv
│
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🔄 End-to-End Workflow

The pipeline begins with data preprocessing and feature engineering, followed by training a machine learning model capable of handling imbalanced data. The trained model is serialized into a `.pkl` file and stored with versioning. The API loads the model and serves real-time predictions. Incoming requests and predictions are logged for monitoring purposes, which can later be used to retrain the model with new data. Retraining produces a new version of the model, continuing the lifecycle.

---

## 🧪 Model Performance

Instead of relying on accuracy (which is misleading for imbalanced datasets), the system evaluates performance using recall, F1-score, and ROC-AUC. Recall is prioritized to ensure fraudulent transactions are not missed, while F1-score ensures a balance between precision and recall.

---

## 🚀 Running Locally

To train the model:

```
python pipeline/train.py
```

To run the API locally:

```
uvicorn app.main:app --reload
```

Access the interactive API documentation at:

```
http://localhost:8000/docs
```

---

## 🐳 Running with Docker

To build the Docker image:

```
docker build -t fraud-api .
```

To run the container:

```
docker run -p 8000:8000 fraud-api
```

---

## 📡 API Endpoints

The system exposes two main endpoints:

* **GET /health** → Returns system status and model info
* **POST /predict** → Accepts transaction data and returns fraud prediction and probability

Example input:

```
{
  "Time": 10000,
  "Amount": 250,
  "V1": -1.2,
  "V2": 0.5
}
```

Example output:

```
{
  "fraud_prediction": 0,
  "fraud_probability": 0.12
}
```

---

## 📊 Logging & Monitoring

The API logs incoming requests and predictions to help track model behavior in production. These logs can be used to detect issues such as data drift and trigger retraining when necessary.

---

## 🔁 Retraining Pipeline

The retraining process combines historical data with newly collected data, retrains the model, and saves a new version of the model file. Versioning ensures traceability and allows rollback if needed.

---

## 🧠 Key Learnings

This project demonstrates handling imbalanced datasets, building production-ready APIs, containerizing machine learning applications, implementing manual model versioning, and designing retraining workflows.

---

## ⚠️ Limitations

The system uses manual model versioning without a centralized registry, has limited monitoring capabilities, and lacks automated CI/CD integration.

---

## 🔥 Future Improvements

Future enhancements include integrating a model registry, deploying to cloud infrastructure, adding CI/CD pipelines, and implementing advanced monitoring such as drift detection.

---

## 👨‍💻 Author

Jainy Bhatt
Aspiring Data Scientist / ML Engineer

---

## ⭐ Summary

A production-oriented fraud detection system that trains, version-controls, deploys, and serves machine learning models through a scalable API, with a complete lifecycle from training to retraining.
