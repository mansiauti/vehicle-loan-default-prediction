# 🚗 Vehicle Loan Default Prediction

Predicts whether a borrower will default on a vehicle loan within 90 days of disbursal — built end-to-end from raw data to a deployed, interactive Streamlit app.

🔗 **Live Demo:** [add your Streamlit Community Cloud link here once deployed]

---

## 📌 Problem Statement

Vehicle loan defaults are a major risk for NBFCs — misjudging risk at disbursal leads to bad debt, while overly strict criteria turns away good customers. This project builds a machine learning pipeline to score default risk at the time of loan application, supporting faster and more consistent underwriting decisions.

---

## 📊 Results

| Model | ROC-AUC | Precision | Recall |
|---|---|---|---|
| Logistic Regression | 0.62 | 0.31 | 0.58 |
| Random Forest | 0.65 | 0.34 | 0.55 |
| **XGBoost (final)** | **0.68** | **0.36** | **0.60** |

**Final model:** XGBoost — F1 Score: 0.45

---

## ⚙️ Pipeline

- **Dataset:** [L&T FinHack (Kaggle)](https://www.kaggle.com/code/poojagupta0710/vehicle-loan-defaulter-prediction) — ~2.33 lakh records, 41 original features
- **EDA & Feature Engineering** — added 6 features: `loan_to_asset_ratio`, `high_inquiry_flag`, `high_ltv_flag`, `total_accounts`, `active_acct_ratio`, `has_overdue`
- **Encoding** — LabelEncoder applied column-by-column
- **Class Imbalance** — SMOTE (dataset has ~21% default rate)
- **Models Trained** — Logistic Regression, Random Forest, XGBoost
- **Explainability** — SHAP values for feature-level interpretation
- **Deployment** — 4-page Streamlit app (Home, Single Prediction, Batch Prediction, Model Insights)

---

## 🧠 Key Decisions

- **ROC-AUC over Accuracy** — with a ~21% default rate, accuracy is misleading; ROC-AUC captures true discriminatory power across all thresholds.
- **XGBoost + SHAP over black-box models** — RBI/NBFC lending regulations favor explainable credit models; SHAP makes individual predictions interpretable for compliance and underwriting teams.
- **SMOTE trade-off** — generates synthetic minority-class samples by interpolating between real examples and their nearest neighbors. This helps the model learn minority-class patterns but can introduce noise — a known limitation worth monitoring in production.

---

## 🛠️ Tech Stack

`Python | XGBoost | Scikit-learn | SMOTE (imbalanced-learn) | SHAP | Streamlit | Pandas | NumPy | Matplotlib | Seaborn`

---

## 📂 Project Structure

```
loan-default-scorecard/
├── README.md
├── requirements.txt
├── .gitignore
├── app.py
├── Notebook/
│   └── loan_default_eda_modeling.ipynb
├── Model/
│   ├── xgb_model.pkl
│   └── columns.pkl
└── sample_batch.csv
```

---

## ▶️ How to Run Locally

```bash
git clone https://github.com/mansiauti/vehicle-loan-default-prediction.git
cd vehicle-loan-default-prediction
pip install -r requirements.txt
streamlit run app.py
```
If the `Model/` files are not present, the app automatically falls back to **Demo Mode** using a rule-based heuristic, so the UI is always explorable.

---

## ⚠️ Limitations

- SMOTE-generated synthetic samples may introduce noise
- Model does not use personally identifying features (name/DOB) — designed with privacy in mind
- Recommended: regular production monitoring (PSI, KS statistic) to track model drift

---

## 👤 Author

**Mansi Auti**
MSc Data Science, University of Mumbai
[LinkedIn] · [GitHub](https://github.com/mansiauti)
