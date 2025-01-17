import streamlit as st
import pandas as pd
import joblib

# Function to load the Random Forest model
def load_model():
    model_file = 'best_random_forest_model.sav'
    model = joblib.load(model_file)
    st.write(f"Model '{model_file}' loaded successfully!")
    return model

# Streamlit UI
st.title('Diabetes Prediction Model')

st.write("""
    Loading the Random Forest model for prediction.
""")

# Input fields for prediction
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

# Load the Random Forest model
model = load_model()

# Make prediction if model is loaded
if st.button('Predict'):
    prediction = model.predict(input_data)
    
    if prediction == 1:
        st.write('Hasil Prediksi: Positif Diabetes')
    else:
        st.write('Hasil Prediksi: Negatif Diabetes')
