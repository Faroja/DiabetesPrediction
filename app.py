import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model
model = joblib.load('random_forest_model.pkl')

# Function to make predictions
def predict_diabetes(data):
    # Predict whether the person has diabetes (1) or not (0)
    prediction = model.predict(data)
    return prediction

# Streamlit UI
st.title('Prediksi Diabetes')

st.write("""
    Masukkan data berikut untuk memprediksi apakah seseorang memiliki diabetes atau tidak.
    """)

# Input fields for user to enter data
glucose = st.number_input('Glucose Level (mg/dL)', min_value=0, max_value=200, step=1)
blood_pressure = st.number_input('Blood Pressure (mm Hg)', min_value=0, max_value=200, step=1)
skin_thickness = st.number_input('Skin Thickness (mm)', min_value=0, max_value=100, step=1)
insulin = st.number_input('Insulin (mu U/ml)', min_value=0, max_value=1000, step=1)
bmi = st.number_input('BMI (kg/m^2)', min_value=0.0, max_value=100.0, step=0.1)
diabetes_pedigree = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, step=0.01)
age = st.number_input('Age (years)', min_value=0, max_value=100, step=1)

# Create DataFrame for prediction
input_data = pd.DataFrame([[glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]], 
                          columns=['GLUCOSE', 'BLOODPRESSURE', 'SKINTHICKNESS', 'INSULIN', 'BMI', 'DIABETESPEDIGREEFUNCTION', 'AGE'])

# Predict when the user presses the button
if st.button('Prediksi'):
    prediction = predict_diabetes(input_data)
    
    if prediction == 1:
        st.write('Hasil Prediksi: Positif Diabetes')
    else:
        st.write('Hasil Prediksi: Negatif Diabetes')
