import pandas as pd
import joblib
from xgboost import XGBClassifier

# Load old + new data
old = pd.read_csv("data/processed_data.csv")
new = pd.read_csv("data/new_data.csv")

df = pd.concat([old,new])

X = df.drop("Class", axis=1)
y = df["Class"]

model = XGBClassifier(objective='binary:logistic', eval_metric='logloss',
                          scale_pos_weight=scale_pos_weight_value, random_state=42)
model.fit(X,y)

#save new model version
joblib.dump(model, "models/fraud_model_v2.pkl")

print("Retraining complete")