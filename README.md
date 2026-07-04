<div align="center">

# 📉 Customer Churn Prediction
### *Predicting who's about to walk out the door — before they do*

[![Live App](https://img.shields.io/badge/🚀_Live_Demo-Streamlit-FF4B4B?style=for-the-badge)](https://customer-churn-prediction-u7fzk6jkexnmijedfdrcbv.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Model-77B900?style=for-the-badge)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP-8A2BE2?style=for-the-badge)](https://shap.readthedocs.io/)

</div>

---

## 🎯 The Problem

Telecom companies bleed revenue every year to customers who quietly leave. Acquiring a new customer costs 5–25x more than keeping an existing one. The question isn't just *"will this customer churn?"* — it's *"why, and what can we do about it before it happens?"*

This project answers both.

## ⚡ What It Does

A full machine learning pipeline that goes from raw customer data → risk score → **human-readable explanation**, wrapped in a live dashboard anyone (technical or not) can use.

> Type in a customer's profile → get a churn risk score → see *exactly* which factors are driving that risk, visually.

🔗 **[Try it live →](https://customer-churn-prediction-u7fzk6jkexnmijedfdrcbv.streamlit.app/)**

## 🔍 What the Data Revealed

| Insight | Detail |
|---|---|
| 📄 Contract type is king | Month-to-month customers churn at **~43%** vs. long-term contracts |
| 🐣 New customers are flight risks | Low-tenure customers churn significantly more |
| 💸 Price sensitivity is real | Higher monthly charges correlate strongly with churn |

## 🛠️ How It's Built
Raw Data → EDA → Feature Engineering → SMOTE (imbalance fix)
→ Model Race (LogReg vs RF vs XGBoost) → SHAP Explainability → Streamlit App

**Feature Engineering highlights:** `ChargesPerTenure`, `IsNewCustomer`, `TotalServices` — custom features designed to capture behavior, not just raw stats.

**Model Race Results (ROC-AUC):**

🥇 XGBoost — best performer (~0.85+)
🥈 Random Forest
🥉 Logistic Regression (baseline)

**Explainability:** Every prediction comes with a SHAP waterfall plot — no black-box "trust me," just a clear breakdown of *why* this specific customer is flagged.

## 🧰 Tech Stack

`Python` · `Pandas` · `Scikit-learn` · `XGBoost` · `SHAP` · `imbalanced-learn` · `Streamlit`

## 📁 Repository Structure
├── app.py                    # The dashboard (Streamlit)
├── churn_prediction.ipynb    # Full EDA + modeling journey
├── xgb_model.pkl             # Trained model
├── scaler.pkl                # Fitted scaler
├── feature_names.pkl         # Feature schema for inference
├── requirements.txt
└── archive/                  # Raw dataset

## 🚀 Run It Yourself

```bash
git clone https://github.com/AkshayEtukuri/customer-churn-prediction.git
cd customer-churn-prediction
pip install -r requirements.txt
streamlit run app.py
```

## 👤 Author

**Akshay Etukuri**
[GitHub](https://github.com/AkshayEtukuri)

---
<div align="center">
<i>Part of an ongoing portfolio applying ML to real business and security problems.</i>
</div>
