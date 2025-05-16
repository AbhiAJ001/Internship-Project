# Import Libraries 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import os

# Load dataset
df = pd.read_csv("data/Telco-Customer-Churn.csv")

# Drop customerID column
df.drop("customerID", axis=1, inplace=True)

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

# Encode categorical columns
categorical_cols = df.select_dtypes(include=["object"]).columns
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Separate features and target
selected_features = ['gender', 'Partner', 'InternetService', 'Contract',
    'tenure', 'MonthlyCharges', 'TotalCharges']
X = df[selected_features]
y = df["Churn"]

# Scale numerical features
scaler = StandardScaler()
num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
X[num_cols] = scaler.fit_transform(X[num_cols])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and preprocessing tools
os.makedirs("model", exist_ok=True)
with open("model/churn_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("model/encoder.pkl", "wb") as f:
    pickle.dump(encoders, f)

print("Model training complete.")
