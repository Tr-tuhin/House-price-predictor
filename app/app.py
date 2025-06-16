# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load("models/model.pkl")

st.set_page_config(page_title="House Price Predictor", page_icon="🏠")
st.title("🏠 House Price Predictor")

st.markdown("Enter the property details below to estimate its selling price.")

# Input form
bedrooms = st.number_input("Bedrooms", 1, 10, 3)
bathrooms = st.number_input("Bathrooms", 1.0, 10.0, 2.0)
sqft_living = st.number_input("Sqft Living", 300, 10000, 2000)
sqft_lot = st.number_input("Sqft Lot", 300, 20000, 5000)
floors = st.number_input("Floors", 1.0, 3.5, 1.0)
waterfront = st.selectbox("Waterfront", [0, 1])
view = st.slider("View", 0, 4, 0)
condition = st.slider("Condition", 1, 5, 3)
sqft_above = st.number_input("Sqft Above", 300, 10000, 1500)
sqft_basement = st.number_input("Sqft Basement", 0, 5000, 500)
yr_built = st.number_input("Year Built", 1900, 2024, 2000)
yr_renovated = st.number_input("Year Renovated", 0, 2024, 0)

# Predict
if st.button("Predict Price"):
    input_data = np.array([[bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront,
                            view, condition, sqft_above, sqft_basement, yr_built, yr_renovated]])
    prediction = model.predict(input_data)[0]
    st.success(f"💲 Estimated House Price: **${prediction:,.2f}**")
