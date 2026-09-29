import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Load saved model and preprocessing objects
model = joblib.load('best_flight_model_pca.pkl')
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca_transformer.pkl')

st.title(' Flight Delay Prediction App')
st.write(
    'Enter flight details below. Inputs are standardized and transformed via'
    ' PCA before prediction.'
)

# User input widgets matching the 8 numerical features used in PCA
sched_dep = st.number_input('Scheduled Departure Time (e.g., 1430)', min_value=0, max_value=2359, value=1200)
dep_delay = st.number_input('Departure Delay (minutes)', min_value=-60, max_value=300, value=0)
taxi_out = st.number_input('Taxi Out Time (minutes)', min_value=0, max_value=120, value=15)
sched_time = st.number_input('Scheduled Time (minutes)', min_value=10, max_value=500, value=120)
elapsed_time = st.number_input('Elapsed Time (minutes)', min_value=10, max_value=500, value=130)
air_time = st.number_input('Air Time (minutes)', min_value=5, max_value=450, value=100)
distance = st.number_input('Flight Distance (miles)', min_value=50, max_value=5000, value=500)
taxi_in = st.number_input('Taxi In Time (minutes)', min_value=0, max_value=120, value=10)

if st.button('Predict Delay'):
  # Create input dataframe
  input_data = pd.DataFrame([[sched_dep, dep_delay, taxi_out, sched_time, elapsed_time, air_time, distance, taxi_in]], 
                            columns=['SCHEDULED_DEPARTURE', 'DEPARTURE_DELAY', 'TAXI_OUT', 'SCHEDULED_TIME', 'ELAPSED_TIME', 'AIR_TIME', 'DISTANCE', 'TAXI_IN'])
  
  # Apply identical preprocessing pipeline
  input_scaled = scaler.transform(input_data)
  input_pca = pca.transform(input_scaled)
  
  prediction = model.predict(input_pca)
  prob = model.predict_proba(input_pca)[0][1]

  if prediction[0] == 1:
    st.error(f' High Likelihood of Delay! (Probability: {prob:.2%})')
  else:
    st.success(f' Flight is likely to be On-Time. (Delay Probability: {prob:.2%})')
