# Customer Churn Prediction

Machine learning project that predicts customer churn using XGBoost, SHAP explainability, and a live Streamlit dashboard.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-orange)

---

## Problem

Businesses lose significant revenue to customer churn. This project builds an end-to-end ML pipeline that predicts which customers are likely to churn, explains *why* using SHAP, and delivers predictions through a live interactive dashboard.

## Dataset

IBM Telco Customer Churn dataset — 7,043 customers, 20 features including contract type, tenure, monthly charges, internet service, and payment method. Target: `Churn Value` (0 = stayed, 1 = churned). Class imbalance handled with SMOTE (26.5% churn rate).

## Key Insights from EDA

- Month-to-month contracts have the highest churn rate (~43%)
- New customers (tenure < 6 months) churn significantly more
- Higher monthly charges correlate strongly with churn

## Feature Engineering

Two new features created that improved model performance:

- `ChargesPerTenure` — average monthly spend per month of tenure (became the #1 SHAP feature)
- `IsNewCustomer` — binary flag for customers with tenure ≤ 6 months

## Models Trained

| Model | AUC Score |
|---|---|
| Logistic Regression | 0.8513 |
| Random Forest | 0.8337 |
| XGBoost | 0.84+ |

Logistic Regression achieved the best AUC. XGBoost used for SHAP explainability.

## SHAP Explainability

Top features driving churn predictions:

1. `ChargesPerTenure` — engineered feature, most predictive
2. `Dependents` — customers with dependents less likely to churn
3. `Contract_Two year` — long-term contracts reduce churn significantly

## Project Structure
customer-churn-prediction/
├── app.py                        # Streamlit dashboard
├── churn_prediction.ipynb        # Full ML pipeline notebook
├── xgb_model.pkl                 # Saved XGBoost model
├── scaler.pkl                    # Saved StandardScaler
├── feature_names.pkl             # Saved feature names
├── archive/
│   └── Telco_customer_churn.xlsx # Dataset
└── README.md
## Running Locally

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn shap streamlit openpyxl
python -m streamlit run app.py
```

## Results

- Correctly identifies HIGH RISK customers (month-to-month, high charges, low tenure)
- Correctly identifies LOW RISK customers (long tenure, two-year contract, lower charges)
- SHAP waterfall plots explain individual predictions

## Author

Akshay Etukuri — [github.com/AkshayEtukuri](https://github.com/AkshayEtukuri)
