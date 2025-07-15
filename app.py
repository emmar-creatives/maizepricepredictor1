import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Load saved scaler and model
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

model = load_model('lstm_model.h5')

st.title(" Maize Price Prediction App")

st.write("Enter the latest data to predict the maize price per husk:")

# Input fields
price = st.number_input("Last Month Price (₦)", value=150)
rainfall = st.number_input("Rainfall (mm)", value=50)
temperature = st.number_input("Temperature (°C)", value=28)
humidity = st.number_input("Humidity (%)", value=60)
fuel_price = st.number_input("Fuel Price (₦/litre)", value=650)
production = st.number_input("Production (Tonnes)", value=1000)
demand = st.number_input("Demand Spike", value=0)

if st.button("Predict Price"):
    # Create input array with last 6 months, but here we fake it with same input repeated
    last_6_months = np.array([[price, rainfall, temperature, humidity, fuel_price, production, demand]] * 6)
    
    # Scale
    scaled_input = scaler.transform(last_6_months)
    
    # Shape to (1, 6, features)
    final_input = np.expand_dims(scaled_input, axis=0)
    
    # Predict
    prediction = model.predict(final_input)
    
    # Inverse scale prediction
    dummy = np.zeros((1, scaled_input.shape[1]))
    dummy[0,0] = prediction
    inv_pred = scaler.inverse_transform(dummy)[0,0]
    
    st.success(f" Predicted Maize Price per Husk: ₦{inv_pred:.2f}")
