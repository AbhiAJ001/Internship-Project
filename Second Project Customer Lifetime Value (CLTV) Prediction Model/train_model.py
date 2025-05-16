# Import Libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Load Data 
df = pd.read_excel("data/Online Retail.xlsx")
df.dropna(subset=['CustomerID'], inplace=True)
df = df[df['Quantity'] > 0]
df = df[df['UnitPrice'] > 0]
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

# Set Snapshot Date
snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

# RFM Feature Engineering
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'TotalPrice': 'sum'
}).reset_index()

rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

# Prepare Features
X = rfm[['Recency', 'Frequency', 'Monetary']]
y = rfm['Monetary']

# Scale Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Model 
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save Model & Scaler
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/cltv_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")

print("Model training complete.")
