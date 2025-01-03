import streamlit as st
import numpy as np
from tensorflow.keras.models import model_from_json
from sklearn.preprocessing import MinMaxScaler
import joblib

# Load the saved model with Joblib
model_structure, model_weights = joblib.load('traffic_prediction_model.pkl')
model = model_from_json(model_structure)
model.set_weights(model_weights)

# MinMaxScaler setup (adjust based on your training data)
scaler = MinMaxScaler()
scaler.fit(np.array([0, 180]).reshape(-1, 1))  # Replace 180 with the max traffic value from your dataset

# Streamlit app
st.title("Traffic Flow Prediction App 🚦")
st.write("Predict vehicle traffic at specific times based on historical trends.")

# User inputs
day_of_week = st.selectbox("Day of the Week (0=Monday, 6=Sunday):", range(7))
hour = st.slider("Hour of the Day (0-23):", 0, 23)
is_holiday = st.selectbox("Is it a holiday? (0=No, 1=Yes):", [0, 1])

# Prepare input for prediction
input_features = np.array([hour, day_of_week, is_holiday, 0]).reshape(1, -1)  # Replace '0' with a placeholder for Vehicles

# Scale only the relevant feature (Vehicles)
vehicles_scaled = scaler.transform([[0]])[0][0]  # Placeholder scaling for Vehicles
input_features[:, 3] = vehicles_scaled  # Replace the placeholder with the scaled value

# Make prediction
if st.button("Predict Traffic"):
    prediction_scaled = model.predict(input_features)
    prediction = scaler.inverse_transform([[prediction_scaled[0][0]]])[0][0]  # Unscale the prediction
    st.write(f"Predicted Traffic Flow: **{int(prediction)} vehicles**")
