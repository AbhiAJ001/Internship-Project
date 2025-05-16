# Import Libraries
import streamlit as st
import joblib
import numpy as np
import os

# Load model
model_path = os.path.join("model", "cltv_model.pkl")
model = joblib.load(model_path)

st.set_page_config(page_title="Customer LTV Prediction")
st.title("Customer Lifetime Value (CLTV) Prediction")

st.sidebar.header("Enter Customer RFM Details")

# Input fields
recency = st.sidebar.number_input("Recency (days since last purchase)", min_value=0, value=30)
frequency = st.sidebar.number_input("Frequency (number of purchases)", min_value=1, value=5)
monetary = st.sidebar.number_input("Monetary (average spending per purchase)", min_value=1.0, value=100.0)

# Predict
if st.sidebar.button("Predict CLTV"):
    input_data = np.array([[recency, frequency, monetary]])
    prediction = model.predict(input_data)
    st.subheader("Predicted Customer Lifetime Value:")
    st.success(f"£{prediction[0]:.2f}")
