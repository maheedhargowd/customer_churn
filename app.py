import streamlit as st
import joblib
import numpy as np

# Load the saved model and scaler
model = joblib.load("churn_prediction_model.pkl")
scaler = joblib.load("scaler.pkl")

# Define a function to make predictions
def predict_churn(features):
    scaled_features = scaler.transform([features])  # Scale user input
    prediction = model.predict(scaled_features)[0]
    probability = model.predict_proba(scaled_features)[0][1]  # Probability of churn
    return prediction, probability

# Streamlit App Layout
st.title("Customer Churn Prediction App")
st.write("Enter customer details to predict the churn probability.")

# User Inputs
gender = st.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.selectbox("Senior Citizen", ["Yes", "No"])
partner = st.selectbox("Has Partner", ["Yes", "No"])
dependents = st.selectbox("Has Dependents", ["Yes", "No"])
tenure = st.slider("Tenure (in months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=150.0, value=50.0)

# Categorical Service Features
phone_service = st.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security", ["Yes", "No"])
online_backup = st.selectbox("Online Backup", ["Yes", "No"])
device_protection = st.selectbox("Device Protection", ["Yes", "No"])
tech_support = st.selectbox("Tech Support", ["Yes", "No"])
streaming_tv = st.selectbox("Streaming TV", ["Yes", "No"])
streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No"])

# Contract & Payment Features
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.selectbox("Payment Method", 
                              ["Electronic check", "Mailed check", 
                               "Bank transfer (automatic)", "Credit card (automatic)"])

# Convert categorical values to numerical
gender = 1 if gender == "Male" else 0
senior_citizen = 1 if senior_citizen == "Yes" else 0
partner = 1 if partner == "Yes" else 0
dependents = 1 if dependents == "Yes" else 0
paperless_billing = 1 if paperless_billing == "Yes" else 0

# Binary categorical features
phone_service = 1 if phone_service == "Yes" else 0
multiple_lines = 1 if multiple_lines == "Yes" else 0
online_security = 1 if online_security == "Yes" else 0
online_backup = 1 if online_backup == "Yes" else 0
device_protection = 1 if device_protection == "Yes" else 0
tech_support = 1 if tech_support == "Yes" else 0
streaming_tv = 1 if streaming_tv == "Yes" else 0
streaming_movies = 1 if streaming_movies == "Yes" else 0

# One-hot encoding for categorical features
internet_service_fiber = 1 if internet_service == "Fiber optic" else 0
internet_service_no = 1 if internet_service == "No" else 0
contract_one_year = 1 if contract == "One year" else 0
contract_two_year = 1 if contract == "Two year" else 0
payment_credit_card = 1 if payment_method == "Credit card (automatic)" else 0
payment_electronic_check = 1 if payment_method == "Electronic check" else 0
payment_mailed_check = 1 if payment_method == "Mailed check" else 0

# Calculate AvgMonthlySpend (Avoid division by zero)
avg_monthly_spend = monthly_charges / tenure if tenure > 0 else monthly_charges

# Combine user inputs (Matching feature order used in training)
features = [
    gender, senior_citizen, partner, dependents, tenure, 
    phone_service, multiple_lines, online_security, online_backup, 
    device_protection, tech_support, streaming_tv, streaming_movies, 
    paperless_billing, monthly_charges, 
    internet_service_fiber, internet_service_no, 
    contract_one_year, contract_two_year, 
    payment_credit_card, payment_electronic_check, payment_mailed_check, 
    avg_monthly_spend
]

# Predict Churn
if st.button("Predict Churn"):
    prediction, probability = predict_churn(features)
    
    if prediction == 1:
        st.error(f"This customer is **likely to churn** (Probability: {probability:.2f})")
    else:
        st.success(f"This customer is **not likely to churn** (Probability: {probability:.2f})")
