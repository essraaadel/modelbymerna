
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- 1. Load Models and Feature Columns ---
@st.cache_resource
def load_models():
    dt_model = joblib.load('decision_tree_model.joblib')
    rf_model = joblib.load('random_forest_model.joblib')
    lgbm_model = joblib.load('lightgbm_model.joblib')
    feature_columns = joblib.load('feature_columns.joblib')
    return dt_model, rf_model, lgbm_model, feature_columns

dt_model, rf_model, lgbm_model, feature_columns = load_models()

# --- 2. Define Feature Specifications for Input Collection ---
# Based on EDA (df.describe(), df.unique(), df.info() outputs)
feature_specs = {
    'Group': {'dtype': int, 'options': [1, 2], 'help': 'Patient group (1 or 2)'},
    'Sex': {'dtype': int, 'options': [1, 2], 'help': 'Patient sex (1 for Male, 2 for Female)'},
    'Age': {'dtype': float, 'min_val': 16.0, 'max_val': 96.0, 'step': 1.0, 'help': 'Age of the patient'},
    'Patients number per hour': {'dtype': float, 'min_val': 1.0, 'max_val': 17.0, 'step': 1.0, 'help': 'Number of patients per hour'},
    'Arrival mode': {'dtype': int, 'options': [1, 2, 3, 4, 5, 6, 7], 'help': 'Mode of arrival (e.g., Ambulance, Walk-in)'},
    'Injury': {'dtype': int, 'options': [1, 2], 'help': 'Injury status (1 or 2)'},
    'Mental': {'dtype': int, 'options': [1, 2, 3, 4], 'help': 'Mental status (1-4)'},
    'Pain': {'dtype': int, 'options': [0, 1], 'help': 'Pain presence (0 for No, 1 for Yes)'},
    'NRS_pain': {'dtype': float, 'min_val': 1.0, 'max_val': 10.0, 'step': 0.1, 'help': 'NRS Pain Scale (0-10)'},
    'SBP': {'dtype': float, 'min_val': 50.0, 'max_val': 275.0, 'step': 0.1, 'help': 'Systolic Blood Pressure'},
    'DBP': {'dtype': float, 'min_val': 31.0, 'max_val': 160.0, 'step': 0.1, 'help': 'Diastolic Blood Pressure'},
    'HR': {'dtype': float, 'min_val': 32.0, 'max_val': 148.0, 'step': 0.1, 'help': 'Heart Rate'},
    'RR': {'dtype': float, 'min_val': 14.0, 'max_val': 30.0, 'step': 0.1, 'help': 'Respiratory Rate'},
    'BT': {'dtype': float, 'min_val': 35.0, 'max_val': 41.0, 'step': 0.1, 'help': 'Body Temperature'},
    'Saturation': {'dtype': float, 'min_val': 20.0, 'max_val': 100.0, 'step': 0.1, 'help': 'Oxygen Saturation'},
    'Disposition': {'dtype': int, 'options': [1, 2, 3, 4, 5, 6, 7], 'help': 'Patient disposition'},
    'KTAS_expert': {'dtype': int, 'options': [1, 2, 3, 4, 5], 'help': 'KTAS level assigned by expert'},
    'Error_group': {'dtype': int, 'options': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 'help': 'Error group for misdiagnosis'},
    'Length of stay_min': {'dtype': float, 'min_val': 0.0, 'max_val': 709510.0, 'step': 1.0, 'help': 'Length of stay in minutes'},
    'KTAS duration_min': {'dtype': float, 'min_val': 1.0, 'max_val': 17.37, 'step': 0.01, 'help': 'KTAS duration in minutes'},
    'mistriage': {'dtype': int, 'options': [0, 1, 2], 'help': 'Mistriage status (0, 1, or 2)'}
}

# --- 3. Streamlit App Layout and Input Collection ---
st.title('KTAS_RN Prediction App')
st.markdown('Enter patient details to predict the KTAS_RN level.')

user_inputs = {}
for feature, specs in feature_specs.items():
    if 'options' in specs:
        user_inputs[feature] = st.selectbox(
            f"{feature}:",
            options=specs['options'],
            help=specs.get('help', '')
        )
    else:
        user_inputs[feature] = st.number_input(
            f"{feature}:",
            min_value=specs.get('min_val'),
            max_value=specs.get('max_val'),
            value=float(np.mean([specs.get('min_val', 0), specs.get('max_val', 1)])), # Default to midpoint
            step=specs.get('step', 1.0),
            help=specs.get('help', '')
        )

# --- 4. Prediction Button ---
if st.button('Predict KTAS_RN'):
    # Convert user inputs to DataFrame
    input_df = pd.DataFrame([user_inputs])

    # Ensure column order matches training data
    input_df = input_df[feature_columns] # Reorder to match the expected feature_columns from training

    st.subheader('Prediction Results:')

    # Predict with Decision Tree Classifier
    dt_prediction = dt_model.predict(input_df)[0]
    st.write(f"Decision Tree Classifier predicts KTAS_RN: **{int(dt_prediction)}**")

    # Predict with Random Forest Classifier
    rf_prediction = rf_model.predict(input_df)[0]
    st.write(f"Random Forest Classifier predicts KTAS_RN: **{int(rf_prediction)}**")

    # Predict with LightGBM Classifier
    lgbm_prediction = lgbm_model.predict(input_df)[0]
    st.write(f"LightGBM Classifier predicts KTAS_RN: **{int(lgbm_prediction)}**")

st.markdown("""
---
*KTAS_RN levels typically range from 1 (resuscitation) to 5 (non-urgent).*
""")

