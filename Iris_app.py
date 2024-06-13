#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st
import pickle
import numpy as np

# Load the trained model, scaler, and encoder
model = pickle.load(open('Iris_app.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
encoder = pickle.load(open('encoder.pkl', 'rb'))

# Define the prediction function
def predict_dispute(features):
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)
    return encoder.inverse_transform(prediction)[0]

# Streamlit UI
st.title("Iris Prediction Model")

st.header("Enter the details to predict if it will be in dispute or not")

# Create input fields for each feature
sepal_length = st.number_input("Sepal Length", min_value=0.0, max_value=10.0, step=0.1)
sepal_width = st.number_input("Sepal Width", min_value=0.0, max_value=10.0, step=0.1)
petal_length = st.number_input("Petal Length", min_value=0.0, max_value=10.0, step=0.1)
petal_width = st.number_input("Petal Width", min_value=0.0, max_value=10.0, step=0.1)

# Button to predict
if st.button("Predict"):
    features = [sepal_length, sepal_width, petal_length, petal_width]
    prediction = predict_dispute(features)
    st.success(f"The predicted class is: {prediction}")

