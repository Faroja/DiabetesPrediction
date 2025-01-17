import streamlit as st
import pandas as pd
import joblib

# Function to try loading models one by one
def try_loading_model():
    models = ['best_random_forest_model.sav', 'best_xgboost_model.sav', 'best_voting_model.sav']
    
    for model_file in models:
        try:
            model = joblib.load(model_file)
            st.write(f"Model '{model_file}' loaded successfully!")
            return model
        except Exception as e:
            st.write(f"Error loading model '{model_file}': {e}")
            continue
    return None

# Streamlit UI
st.title('Diabetes Prediction Model')

st.write("""
    Attempting to load the first available model from a list.
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

# Try loading the model
model = try_loading_model()

# Make prediction if model is loaded
if model is not None:
    if st.button('Predict'):
        prediction = model.predict(input_data)
        
        if prediction == 1:
            st.write('Hasil Prediksi: Positif Diabetes')
        else:
            st.write('Hasil Prediksi: Negatif Diabetes')
else:
    st.write("No model was loaded. Please check your models.")
