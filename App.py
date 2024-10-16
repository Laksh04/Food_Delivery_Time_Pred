import streamlit as st
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras 
import sys
sys.stdout.reconfigure(encoding='utf-8')



# Load the pickled model and scaler
model = pickle.load(open('FDTP_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title("Food Delivery Time Prediction")

# Get user input
st.header("Input details for prediction")
age = st.number_input("Age of Delivery Person", min_value=18, max_value=70, value=25)
rating = st.number_input("Customer Rating for Last Order", min_value=1.0, max_value=5.0, step=0.1, value=4.0)
distance = st.number_input("Distance of Current Order (km)", min_value=0.1, max_value=100.0, value=5.0)

# Create input variable 'abc' based on the user inputs
abc = np.array([[age, rating, distance]])

# Scale the input if necessary
scaled_abc = scaler.transform(abc)

# Predict the time taken
predicted_time = model.predict(scaled_abc)[0]

# Display the predicted time taken
st.subheader("Predicted Delivery Time:")
st.write(f"The predicted time for delivery is: {predicted_time[0]:.2f} minutes")

