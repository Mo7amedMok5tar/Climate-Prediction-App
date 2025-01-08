import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import datetime

# Load the scaler and model
scaler = joblib.load("E:/My_journy/ML With Hesham Asem/Projects/Cliamte predection/app/scaler.pkl")
model = load_model("E:/My_journy/ML With Hesham Asem/Projects/Cliamte predection/app/Claimate_pred4.h5")

# Define weather categories and their encoded values
weather_order = {
    'sun': 0,       # Least severe weather condition
    'fog': 1,       # Slightly severe
    'drizzle': 2,   # Moderate severity
    'rain': 3,      # High severity
    'snow': 4       # Highest severity
}

# Reverse mapping for weather categories
weather_choices = list(weather_order.keys())

# App title
st.title("Climate Prediction App")

# User selects a date, and we extract day, month, year, and quarter
date_input = st.date_input("Select a date", value=datetime.date.today())
day = date_input.day
month = date_input.month
year = date_input.year
quarter = (month - 1) // 3 + 1  # Calculate the quarter

# Create sliders for user input
precipitation = st.slider("Precipitation", min_value=0.0, max_value=50.0, step=0.1)
temp_max = st.slider("Max Temperature (°C)", min_value=-10.0, max_value=50.0, step=0.1)
temp_min = st.slider("Min Temperature (°C)", min_value=-10.0, max_value=50.0, step=0.1)
wind = st.slider("Wind Speed (km/h)", min_value=0.0, max_value=100.0, step=0.5)


# Calculate derived features
avg_temp = (temp_max + temp_min) / 2
rang_temp = temp_max - temp_min
is_heatwave = 1 if temp_min >= 18 else 0
is_frost = 1 if temp_min < 7 else 0

# User selects previous weather condition, which is converted to its encoded value
weather_lag_1_choice = st.selectbox("Previous day's weather", options=weather_choices)
weather_lag_1 = weather_order[weather_lag_1_choice]

# Input lag values manually
temp_max_lag_1 = st.slider("Max Temperature Lag (°C)", min_value=-10.0, max_value=50.0, step=0.1)
precipitation_lag_1 = st.slider("Precipitation Lag", min_value=0.0, max_value=50.0, step=0.1)

# Calculate interaction features
temp_precip_interaction = precipitation * temp_max
precip_wind_interaction = precipitation * wind

# Create DataFrame with input data
data = pd.DataFrame({
    'precipitation': [precipitation],
    'temp_max': [temp_max],
    'temp_min': [temp_min],
    'wind': [wind],
    'day': [day],
    'month': [month],
    'year': [year],
    'quarter': [quarter],
    'humidity': [100 * (np.exp((17.625 * temp_min) / (243.04 + temp_min)) / np.exp((17.625 * temp_max) / (243.04 + temp_max)))],
    'temp_precip_interaction': [temp_precip_interaction],
    'precip_wind_interaction': [precip_wind_interaction],
    'is_frost': [is_frost],
    'is_heatwave': [is_heatwave],
    'temp_max_lag_1': [temp_max_lag_1],
    'avg_temp': [avg_temp],
    'rang_temp': [rang_temp],
    'precipitation_lag_1': [precipitation_lag_1],
    'weather_lag_1': [weather_lag_1]
})

# Normalize the input data using the same scaler used during training
normalized_data = scaler.transform(data)

# Predict probabilities for each weather category
predictions = model.predict(normalized_data)

# Find the index of the highest probability
predicted_index = np.argmax(predictions)

# Get the corresponding weather category
predicted_weather = weather_choices[predicted_index]

# Display the result
st.subheader("Prediction Result:")
st.write(f"The predicted weather condition is: **{predicted_weather.capitalize()}**")
