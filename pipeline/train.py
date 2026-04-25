import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os
from sklearn.pipeline import Pipeline


# Load data
df = pd.read_csv("data/processed_data.csv")

# Split features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# Train-test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


scale_pos_weight_value = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

xgb_model = XGBClassifier(objective='binary:logistic', eval_metric='logloss',
                          scale_pos_weight=scale_pos_weight_value, random_state=42)

# Train the model
xgb_model.fit(X_train, y_train)

# Save model
joblib.dump(xgb_model, "models/fraud_model.pkl")


# Save test data for evaluation
X_test.to_csv("X_test.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("✅ Training complete. Model saved to fraud_model.pkl")


# Define the preprocessing steps and the model
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Step 1: Scale the features
    ('xgb_model', joblib.load("models/fraud_model.pkl")) # Step 2: XGBoost Classifier
])

# Fit the pipeline to the training data
pipeline.fit(X_train, y_train)

print("Pipeline fitted on columns:")
print(X_train.columns.tolist())

# Save the pipeline to a file
joblib.dump(pipeline, "fraud_detection_pipeline.pkl")
print("Pipeline saved successfully.")
scaler = StandardScaler()
scaler.fit(X_train)
joblib.dump(scaler, "models/scaler.pkl")