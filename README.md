# Customer Churn Prediction

An end-to-end machine learning project that predicts customer churn risk for a telecom company, with SHAP-based explainability and a live interactive dashboard.

🔗 **Live Demo:** [customer-churn-prediction-u7fzk6jkexnmijedfdrcbv.streamlit.app](https://customer-churn-prediction-u7fzk6jkexnmijedfdrcbv.streamlit.app/)

## Overview

Telecom companies lose significant revenue to customer churn. This project builds a full pipeline — from raw data to a deployed, explainable prediction app — to identify customers at risk of leaving and explain *why*, so retention teams can act on it.

## Dataset

[IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — 7,043 customers, 21 features (tenure, contract type, monthly charges, internet service, etc.)

## Approach

1. **EDA** — uncovered key churn drivers: month-to-month contracts churn ~43% vs. long-term contracts; new customers (low tenure) churn more; higher monthly charges correlate with higher churn risk.
2. **Feature Engineering** — added `ChargesPerTenure`, `IsNewCustomer`, `TotalServices` to capture behavior beyond raw columns.
3. **Class Imbalance** — handled with SMOTE (applied only to training data, never test data).
4. **Modeling** — compared Logistic Regression, Random Forest, and XGBoost using ROC-AUC; XGBoost performed best.
5. **Explainability** — used SHAP (TreeExplainer) to generate global feature importance and per-customer waterfall plots explaining individual predictions.
6. **Deployment** — built an interactive Streamlit dashboard for real-time churn risk prediction with live SHAP explanations, deployed on Streamlit Community Cloud.

## Tech Stack

`Python` · `Pandas` · `Scikit-learn` · `XGBoost` · `SHAP` · `imbalanced-learn (SMOTE)` · `Streamlit`

## Repository Structure
├── app.py                    # Streamlit dashboard
├── churn_prediction.ipynb    # Full EDA + modeling notebook
├── xgb_model.pkl             # Trained XGBoost model
├── scaler.pkl                # Fitted StandardScaler
├── feature_names.pkl         # Feature column order for inference
├── requirements.txt          # Dependencies
└── archive/                  # Raw dataset

## Run Locally

```bash
git clone https://github.com/AkshayEtukuri/customer-churn-prediction.git
cd customer-churn-prediction
pip install -r requirements.txt
streamlit run app.py
```

## Key Results

- Best model: **XGBoost**, ROC-AUC ~0.85+
- Top churn drivers identified via SHAP: contract type, tenure, total charges

## Author

**Akshay Etukuri** — [GitHub](https://github.com/AkshayEtukuri)
