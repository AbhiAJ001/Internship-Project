
import streamlit as st
import numpy as np
import pickle
import os

# === Load model and preprocessing tools ===
model_path = os.path.join("model", "churn_model.pkl")
scaler_path = os.path.join("model", "scaler.pkl")
encoder_path = os.path.join("model", "encoder.pkl")

model = pickle.load(open(model_path, "rb"))
scaler = pickle.load(open(scaler_path, "rb"))
encoders = pickle.load(open(encoder_path, "rb"))

# === Streamlit UI ===
st.set_page_config(page_title="Customer Churn Prediction")
st.title("Customer Churn Prediction App")
st.sidebar.header("Enter Customer Details")

# === Collect user inputs ===
gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
internet = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly = st.sidebar.number_input("Monthly Charges", 0.0, 200.0, 70.0)
total = st.sidebar.number_input("Total Charges", 0.0, 10000.0, 1500.0)

# === Prediction logic ===
if st.sidebar.button("Predict"):
    try:
        # Encode categorical fields
        input_data = [
            encoders['gender'].transform([gender])[0],
            encoders['Partner'].transform([partner])[0],
            encoders['InternetService'].transform([internet])[0],
            encoders['Contract'].transform([contract])[0],
            tenure,
            monthly,
            total
        ]

        # Scale numerical features
        input_data[-3:] = scaler.transform([input_data[-3:]])[0]

        # Predict
        prediction = model.predict([input_data])[0]
        probability = model.predict_proba([input_data])[0][1]

        # Show result
        st.subheader(" Prediction Result")
        if prediction == 1:
            st.error(f"Customer is likely to churn (Probability: {probability:.2%})")
        else:
            st.success(f" Customer is not likely to churn (Probability: {probability:.2%})")

    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
