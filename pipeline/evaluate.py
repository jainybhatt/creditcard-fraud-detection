import pandas as pd
import joblib
from sklearn.metrics import (classification_report, roc_auc_score, roc_curve, auc)

# Load model
model = joblib.load("fraud_model.pkl")

# Load test data
X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv")


pipeline = joblib.load('models/fraud_detection_pipeline.pkl')

# Make predictions with the pipeline
y_pred_pipeline = pipeline.predict(X_test)
y_probs_pipeline = pipeline.predict_proba(X_test)[:, 1]

# Evaluate the pipeline
print("Classification Report for Pipeline:")
print(classification_report(y_test, y_pred_pipeline))
print("ROC-AUC Score for Pipeline:")
print(roc_auc_score(y_test, y_probs_pipeline))

# Calculate FPR, TPR, and thresholds for the pipeline output
fpr_pipeline, tpr_pipeline, thresholds_pipeline = roc_curve(y_test, y_probs_pipeline)
roc_auc_pipeline = auc(fpr_pipeline, tpr_pipeline)
