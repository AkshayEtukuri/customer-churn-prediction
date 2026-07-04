import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt

# Load saved files
model = pickle.load(open('xgb_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
feature_names = pickle.load(open('feature_names.pkl', 'rb'))

st.set_page_config(page_title="Customer Churn Predictor", layout="wide")
st.title("🔮 Customer Churn Predictor")
st.markdown("Fill in customer details to predict churn risk.")

# Input fields
col1, col2, col3 = st.columns(3)

with col1:
    tenure = st.slider("Tenure (Months)", 0, 72, 12)
    monthly_charges = st.number_input("Monthly Charges ($)", 20.0, 120.0, 65.0)
    total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 500.0)

with col2:
    contract = st.selectbox("Contract Type", 
                            ['Month-to-month', 'One year', 'Two year'])
    internet = st.selectbox("Internet Service", 
                            ['DSL', 'Fiber optic', 'No'])
    payment = st.selectbox("Payment Method", 
                           ['Electronic check', 'Mailed check',
                            'Bank transfer (automatic)', 
                            'Credit card (automatic)'])

with col3:
    partner = st.selectbox("Partner", ['Yes', 'No'])
    dependents = st.selectbox("Dependents", ['Yes', 'No'])
    senior = st.selectbox("Senior Citizen", ['Yes', 'No'])

if st.button("Predict Churn Risk", type="primary"):

    # Build input row with all zeros
    input_dict = {col: 0 for col in feature_names}

    # Fill in numeric values
    input_dict['Tenure Months'] = tenure
    input_dict['Monthly Charges'] = monthly_charges
    input_dict['Total Charges'] = total_charges
    input_dict['ChargesPerTenure'] = monthly_charges / (tenure + 1)
    input_dict['IsNewCustomer'] = 1 if tenure <= 6 else 0
    input_dict['Partner'] = 1 if partner == 'Yes' else 0
    input_dict['Dependents'] = 1 if dependents == 'Yes' else 0
    input_dict['Senior Citizen'] = 1 if senior == 'Yes' else 0

    # Contract
    if contract == 'One year':
        input_dict['Contract_One year'] = 1
    elif contract == 'Two year':
        input_dict['Contract_Two year'] = 1

    # Internet Service
    if internet == 'Fiber optic':
        input_dict['Internet Service_Fiber optic'] = 1
    elif internet == 'No':
        input_dict['Internet Service_No'] = 1

    # Payment Method
    if payment == 'Credit card (automatic)':
        input_dict['Payment Method_Credit card (automatic)'] = 1
    elif payment == 'Electronic check':
        input_dict['Payment Method_Electronic check'] = 1
    elif payment == 'Mailed check':
        input_dict['Payment Method_Mailed check'] = 1

    # Create dataframe
    input_df = pd.DataFrame([input_dict])

    # Scale numeric columns
    numeric = ['Tenure Months', 'Monthly Charges', 
               'Total Charges', 'ChargesPerTenure']
    input_df[numeric] = scaler.transform(input_df[numeric])

    # Predict
    prob = model.predict_proba(input_df)[0][1]

    # Show result
    st.markdown("---")
    if prob > 0.6:
        st.error(f"🔴 HIGH RISK — Churn Probability: {prob:.1%}")
    elif prob > 0.35:
        st.warning(f"🟡 MEDIUM RISK — Churn Probability: {prob:.1%}")
    else:
        st.success(f"🟢 LOW RISK — Churn Probability: {prob:.1%}")

    st.progress(float(prob))

    # SHAP explanation
    st.markdown("### Why this prediction?")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, input_df,
                      feature_names=feature_names,
                      plot_type='bar', show=False)
    st.pyplot(fig)