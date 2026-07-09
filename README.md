
Readme · MD
# 🇲🇦 Credit Card Default Risk Prediction - Moroccan Banking Portfolio
 
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-FF6600?style=for-the-badge&logo=xgboost&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Live_Demo-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-27AE60?style=for-the-badge)
 
> **Production-ready end-to-end ML pipeline** · 30,000 customers · 7 models compared · XGBoost final model · SHAP explainability · Interactive Streamlit dashboard
 
---
 
## 🔗 Links
 
👉 **[Launch Live Dashboard](https://credit-risk-morocco-lb5g69tibchzjlhx3qppqv.streamlit.app)**  
👉 **[View Notebook](notebooks/01_credit_risk_analysis.ipynb)**  
👉 **[Executive Report](reports/final_report.md)**
 
---
 
## 📌 Business Problem
 
Moroccan banks lose **hundreds of millions of MAD annually** to credit card defaults.  
This project builds a machine learning system that predicts which customers will default **next month** - enabling risk teams to intervene early through:
 
- 💳 Credit limit adjustments
- 📋 Structured repayment plan offers
- 🚨 Escalation to manual review
- 📊 Portfolio-level risk monitoring
**Estimated business impact on a 50,000-card portfolio: ~300M MAD saved annually**
 
---
 
## 🖥️ Dashboard Preview
 
| Feature | Description |
|---|---|
| 🎯 Risk Gauge | Live default probability (0–100%) with color-coded tiers |
| 🟢🟡🔴 Risk Tiers | Low / Medium / High with branch manager recommendations |
| 🔍 Key Risk Drivers | SHAP-based feature contribution bar chart |
| 📈 Balance Trend | 6-month statement vs payment chart |
| 🏦 Decision Framework | Full 3-tier approval guidelines |
| 👤 Sample Profiles | Pre-loaded safe / borderline / high-risk test customers |
 
> Branch managers enter customer data → model predicts default probability in real time
 
---
 
## 📊 Results
 
| Model | AUC-ROC | F1 Score | Recall | Precision |
|---|---|---|---|---|
| Logistic Regression | 0.679 | 0.445 | 0.610 | 0.350 |
| K-Nearest Neighbors | 0.643 | 0.190 | 0.121 | 0.435 |
| Decision Tree | 0.660 | 0.435 | 0.572 | 0.351 |
| Random Forest | 0.684 | 0.433 | 0.518 | 0.371 |
| LightGBM | 0.668 | 0.433 | 0.544 | 0.359 |
| CatBoost | 0.684 | 0.445 | 0.585 | 0.359 |
| **✅ XGBoost (Tuned)** | **0.689** | **0.454** | **0.646** | **0.350** |
 
> XGBoost selected as final model · Tuned via RandomizedSearchCV (40 iterations, 5-fold CV)  
> Optimal classification threshold: **0.51** (F1-optimised)
 
---
 
## 🔍 Top 5 Risk Factors (SHAP Analysis)
 
| Rank | Feature | Business Meaning |
|---|---|---|
| 1 | `payment_status_month_1` | Most recent payment delay - strongest single predictor |
| 2 | `credit_utilization` | Avg balance / credit limit - customers > 80% are 2× more likely to default |
| 3 | `max_payment_delay` | Worst-ever delinquency across 6 months |
| 4 | `months_delayed` | How many months had any delay - breadth of risk |
| 5 | `credit_limit_mad` | Inverse relationship - higher limits = lower risk (bank screening effect) |
 
---
 
## 🏦 Decision Framework
 
```
Default Probability    Action
──────────────────────────────────────────────────────────────
0%  – 30%   🟢 LOW RISK     → Approve with standard terms
30% – 60%   🟡 MEDIUM RISK  → Approve with reduced limit + monitoring
60%+        🔴 HIGH RISK    → Decline / Escalate to manual review
```
 
---
 
## 📁 Project Structure
 
```
credit-risk-morocco/
│
├── 📓 notebooks/
│   └── 01_credit_risk_analysis.ipynb    # Full pipeline - 50 cells, 12 steps
│
├── 🐍 src/
│   ├── data_loader.py    # Load, rename columns, TWD → MAD conversion, clean
│   ├── features.py       # 7 engineered features + preprocessing pipeline
│   ├── models.py         # All 7 models + XGBoost hyperparameter tuning
│   └── evaluate.py       # Metrics, charts, SHAP helpers
│
├── 🤖 models/
│   ├── xgboost_final_model.pkl    # Trained best model (serialised)
│   └── preprocessor.pkl           # Fitted ColumnTransformer
│
├── 🌐 app/
│   ├── app.py             # Streamlit dashboard
│   └── requirements.txt
│
├── 📊 data/
│   ├── raw/               # UCI dataset adapted for Morocco (30,000 rows)
│   └── processed/         # Cleaned + feature-engineered CSV
│
├── 📈 reports/
│   ├── figures/           # 17 charts (PNG, 150 DPI)
│   └── final_report.md    # Executive summary for bank leadership
│
├── README.md
└── requirements.txt
```
 
---
 
## ⚙️ Full Pipeline
 
```
Raw Data (30,000 rows · 24 features)
        ↓
    Data Cleaning
    • Column renaming → Moroccan banking terminology
    • TWD → MAD currency conversion (× 0.35)
    • Outlier capping (P99) · Categorical encoding
        ↓
    Feature Engineering (7 new features)
    • credit_utilization · payment_to_bill_ratio
    • months_delayed · max_payment_delay · avg_bill_6m
        ↓
    Stratified Train / Test Split (80 / 20)
        ↓
    Preprocessing (StandardScaler + OrdinalEncoder)
        ↓
    7 Models Trained & Compared
        ↓
    XGBoost Hyperparameter Tuning
    • RandomizedSearchCV · 40 iterations · 5-fold CV
        ↓
    SHAP Explainability + Threshold Tuning
        ↓
    Streamlit Dashboard Deployment
```
 
---
 
## 🚀 Quick Start
 
### 1. Clone & install
 
```bash
git clone https://github.com/MouadShl/credit-risk-morocco.git
cd credit-risk-morocco
 
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux
 
pip install -r requirements.txt
```
 
### 2. Run the notebook
 
```bash
cd notebooks
jupyter notebook 01_credit_risk_analysis.ipynb
```
 
Run all cells - takes ~5–8 minutes. Trains all 7 models and saves outputs.
 
### 3. Launch the dashboard locally
 
```bash
cd app
streamlit run app.py
```
 
Opens at `http://localhost:8501`
 
---
 
## 🛠️ Tech Stack
 
| Category | Tools |
|---|---|
| **Data** | Pandas · NumPy |
| **ML Models** | Scikit-learn · XGBoost · LightGBM · CatBoost |
| **Explainability** | SHAP |
| **Visualisation** | Matplotlib · Seaborn · Plotly |
| **Dashboard** | Streamlit |
| **Serialisation** | Joblib |
| **Environment** | Jupyter · Python 3.10+ |
 
---
 
## 💼 Business Impact
 
```
Portfolio         : 50,000 active credit cards
Loss per default  : 8,500 MAD average
 
Expected monthly defaults    : ~11,350 customers
Defaults caught by model     : ~7,332  customers  (recall 64.6%)
Preventable with intervention: ~2,933  customers  (40% success rate)
 
Monthly losses prevented     : ~24.9M MAD
Annual savings estimate      : ~299M MAD
```
 
---
 
## 📋 Regulatory Compliance
 
- ✅ **SHAP explainability** - every prediction backed by feature-level explanations
- ✅ **Fairness audit** - gender confirmed as < 2% SHAP contribution (not a primary driver)
- ✅ **Reproducible pipeline** - version-controlled, fully documented
- ✅ **SR 11-7 aligned** - model risk management documentation included
- ✅ **Bank Al-Maghrib** - business narrative framed for BAM risk guidelines
---
 
## 👤 Author
 
**Mouad Shl**  
📧 mouadatlas5@gmail.com  
🐙 [github.com/MouadShl](https://github.com/MouadShl)
 
---
 
## 📄 License
 
MIT License - free for personal and commercial use.
 
---
 
*Built as a production-grade data science portfolio project demonstrating end-to-end ML engineering for the banking and fintech domain.*
