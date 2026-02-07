import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import os
import pandas as pd

BASE_DIR = os.path.dirname(__file__)
df = os.path.join(BASE_DIR, "Housing.csv")

st.title("🏠 House Price Prediction App Using Streamlit")

df = pd.read_csv(data_path)

# Convert yes/no to 1/0
df.replace({'yes': 1, 'no': 0}, inplace=True)

# Features & target
X = df[['area', 'bedrooms', 'bathrooms', 'stories', 'parking']]
y = df['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

st.subheader("Enter House Details")

# User inputs
area = st.number_input("Area (sq ft)", min_value=100, max_value=10000, step=100)
bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, step=1)
bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, step=1)
stories = st.number_input("Stories", min_value=1, max_value=5, step=1)
parking = st.number_input("Parking Spaces", min_value=0, max_value=5, step=1)

# Prediction button
if st.button("Predict Price"):
    input_data = np.array([[area, bedrooms, bathrooms, stories, parking]])
    prediction = model.predict(input_data)
    st.success(f"💰 Predicted House Price: ₹ {int(prediction[0]):,}")


