"""
app.py — Moroccan Credit Card Default Risk Predictor
Run with: streamlit run app.py
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk Morocco",
    page_icon="🇲🇦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem; font-weight: 800;
        background: linear-gradient(90deg, #C0392B, #2980B9);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .risk-card {
        border-radius: 12px; padding: 18px 22px; margin: 8px 0;
        border-left: 5px solid;
    }
    .low-risk    { background:#eafaf1; border-color:#27AE60; }
    .medium-risk { background:#fef9e7; border-color:#F39C12; }
    .high-risk   { background:#fdedec; border-color:#C0392B; }
    .metric-box  {
        background:#f8f9fa; border-radius:10px;
        padding:14px 18px; text-align:center; margin:4px;
    }
    .metric-val  { font-size:1.7rem; font-weight:700; }
    .metric-lbl  { font-size:0.82rem; color:#666; }
    div[data-testid="stSidebar"] { background: #1a1a2e; }
    div[data-testid="stSidebar"] * { color: #eee !important; }
</style>
""", unsafe_allow_html=True)

# ── Model Loading ─────────────────────────────────────────────────────────────
MODELS_DIR = Path(__file__).parent.parent / "models"

@st.cache_resource(show_spinner="Loading AI model…")
def load_artifacts():
    model_path = MODELS_DIR / "xgboost_final_model.pkl"
    prep_path  = MODELS_DIR / "preprocessor.pkl"
    if not model_path.exists() or not prep_path.exists():
        return None, None
    return joblib.load(model_path), joblib.load(prep_path)

model, preprocessor = load_artifacts()

# ── Feature helpers ───────────────────────────────────────────────────────────
BILL_COLS    = [f"bill_amount_month_{i}"    for i in range(1, 7)]
PAY_AMT_COLS = [f"payment_amount_month_{i}" for i in range(1, 7)]
PAY_STS_COLS = [f"payment_status_month_{i}" for i in range(1, 7)]

def engineer(row: dict) -> dict:
    bills    = [row[c] for c in BILL_COLS]
    pamts    = [row[c] for c in PAY_AMT_COLS]
    pstatus  = [row[c] for c in PAY_STS_COLS]
    total_b  = sum(bills) or 1
    total_p  = sum(pamts)
    avg_b    = np.mean(bills)
    row["avg_bill_6m"]           = avg_b
    row["avg_payment_6m"]        = np.mean(pamts)
    row["payment_to_bill_ratio"] = min(total_p / total_b, 5.0)
    row["max_payment_delay"]     = max(pstatus)
    row["credit_utilization"]    = min(avg_b / (row["credit_limit_mad"] or 1), 3.0)
    row["payment_consistency"]   = float(np.std(pamts))
    row["months_delayed"]        = sum(1 for s in pstatus if s > 0)
    return row

def build_input_df(inputs: dict) -> pd.DataFrame:
    row = engineer(inputs.copy())
    numeric = (
        ["credit_limit_mad", "age"] + BILL_COLS + PAY_AMT_COLS
        + ["avg_bill_6m", "avg_payment_6m", "payment_to_bill_ratio",
           "max_payment_delay", "credit_utilization",
           "payment_consistency", "months_delayed"]
    )
    cat_ordinal = PAY_STS_COLS
    cat_nominal = ["gender", "education_level", "marital_status"]
    cols = numeric + cat_ordinal + cat_nominal
    return pd.DataFrame([{c: row[c] for c in cols}])

def predict(inputs: dict):
    df = build_input_df(inputs)
    df_proc = preprocessor.transform(df)
    prob = float(model.predict_proba(df_proc)[0, 1])
    return prob

def risk_label(prob: float):
    if prob < 0.30:
        return "🟢 LOW RISK", "low-risk", "#27AE60", "Approve — Standard Terms"
    elif prob < 0.60:
        return "🟡 MEDIUM RISK", "medium-risk", "#F39C12", "Approve — Reduced Limit & Enhanced Monitoring"
    else:
        return "🔴 HIGH RISK", "high-risk", "#C0392B", "Decline / Escalate to Manual Review"

# ── SAMPLE PROFILES ──────────────────────────────────────────────────────────
SAMPLE_PROFILES = {
    "👤 Safe Customer — Salaried Professional": {
        "age": 38, "gender": 1, "education_level": 1, "marital_status": 1,
        "credit_limit_mad": 105_000,
        "bill_amount_month_1": 12_000, "bill_amount_month_2": 11_500,
        "bill_amount_month_3": 10_800, "bill_amount_month_4": 11_200,
        "bill_amount_month_5": 10_500, "bill_amount_month_6": 9_800,
        "payment_amount_month_1": 12_000, "payment_amount_month_2": 11_500,
        "payment_amount_month_3": 10_800, "payment_amount_month_4": 11_200,
        "payment_amount_month_5": 10_500, "payment_amount_month_6": 9_800,
        "payment_status_month_1": -1, "payment_status_month_2": -1,
        "payment_status_month_3": -1, "payment_status_month_4": -1,
        "payment_status_month_5": -1, "payment_status_month_6": -1,
    },
    "⚠️  Borderline — Self-Employed, Occasional Delays": {
        "age": 32, "gender": 0, "education_level": 3, "marital_status": 2,
        "credit_limit_mad": 35_000,
        "bill_amount_month_1": 28_000, "bill_amount_month_2": 26_500,
        "bill_amount_month_3": 24_000, "bill_amount_month_4": 22_000,
        "bill_amount_month_5": 20_500, "bill_amount_month_6": 19_000,
        "payment_amount_month_1": 1_500, "payment_amount_month_2": 2_000,
        "payment_amount_month_3": 1_800, "payment_amount_month_4": 2_500,
        "payment_amount_month_5": 1_200, "payment_amount_month_6": 1_000,
        "payment_status_month_1": 1, "payment_status_month_2": 0,
        "payment_status_month_3": 1, "payment_status_month_4": 0,
        "payment_status_month_5": 0, "payment_status_month_6": -1,
    },
    "🔴 High Risk — Chronic Delinquent": {
        "age": 26, "gender": 0, "education_level": 4, "marital_status": 2,
        "credit_limit_mad": 14_000,
        "bill_amount_month_1": 13_500, "bill_amount_month_2": 13_200,
        "bill_amount_month_3": 12_800, "bill_amount_month_4": 12_500,
        "bill_amount_month_5": 12_000, "bill_amount_month_6": 11_500,
        "payment_amount_month_1": 0, "payment_amount_month_2": 500,
        "payment_amount_month_3": 0, "payment_amount_month_4": 0,
        "payment_amount_month_5": 300, "payment_amount_month_6": 0,
        "payment_status_month_1": 3, "payment_status_month_2": 2,
        "payment_status_month_3": 2, "payment_status_month_4": 1,
        "payment_status_month_5": 1, "payment_status_month_6": 0,
    },
}

# ── SESSION STATE ─────────────────────────────────────────────────────────────
DEFAULTS = SAMPLE_PROFILES["👤 Safe Customer — Salaried Professional"]
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">🇲🇦 Moroccan Credit Card Default Risk Predictor</p>',
            unsafe_allow_html=True)
st.markdown("**AI-powered early warning system** · XGBoost model trained on 30,000 customer accounts · "
            "SHAP-explainable for Bank Al-Maghrib compliance")
st.divider()

if model is None:
    st.error("⚠️  Model not found. Please run the notebook first to train and save the model.")
    st.code("# In the notebooks/ folder, run all cells of 01_credit_risk_analysis.ipynb")
    st.stop()

# ── SIDEBAR — INPUT FORM ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📋 Customer Profile")

    # Sample loader
    sample_choice = st.selectbox("Load Sample Profile", ["— enter manually —"] + list(SAMPLE_PROFILES))
    if sample_choice != "— enter manually —":
        for k, v in SAMPLE_PROFILES[sample_choice].items():
            st.session_state[k] = v

    st.markdown("### 👤 Demographics")
    age = st.slider("Age", 21, 75, key="age")
    gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Male" if x == 0 else "Female",
                          key="gender")
    education_level = st.selectbox(
        "Education Level",
        [1, 2, 3, 4],
        format_func=lambda x: {1: "Graduate", 2: "University", 3: "High School", 4: "Other"}[x],
        key="education_level",
    )
    marital_status = st.selectbox(
        "Marital Status",
        [1, 2, 3],
        format_func=lambda x: {1: "Married", 2: "Single", 3: "Other"}[x],
        key="marital_status",
    )

    st.markdown("### 💳 Credit Profile")
    credit_limit_mad = st.slider("Credit Limit (MAD)", 3_500, 350_000, step=500,
                                  key="credit_limit_mad")

    st.markdown("### 📅 Statement Balances (MAD)")
    col_hint = "Month 1 = most recent"
    st.caption(col_hint)
    for i in range(1, 7):
        st.number_input(f"Balance M{i}", 0, 500_000, step=500,
                        key=f"bill_amount_month_{i}")

    st.markdown("### 💰 Payments Made (MAD)")
    for i in range(1, 7):
        st.number_input(f"Payment M{i}", 0, 500_000, step=100,
                        key=f"payment_amount_month_{i}")

    st.markdown("### ⏱ Payment Status")
    st.caption("-1=Paid in full | 0=Min paid | 1–8=Months delayed")
    for i in range(1, 7):
        st.slider(f"Status M{i}", -2, 8, key=f"payment_status_month_{i}")

    predict_btn = st.button("🔍 Predict Default Risk", type="primary", use_container_width=True)

# ── MAIN PANEL ────────────────────────────────────────────────────────────────
inputs = {
    "age": st.session_state.age,
    "gender": st.session_state.gender,
    "education_level": st.session_state.education_level,
    "marital_status": st.session_state.marital_status,
    "credit_limit_mad": st.session_state.credit_limit_mad,
    **{f"bill_amount_month_{i}":    st.session_state[f"bill_amount_month_{i}"]    for i in range(1, 7)},
    **{f"payment_amount_month_{i}": st.session_state[f"payment_amount_month_{i}"] for i in range(1, 7)},
    **{f"payment_status_month_{i}": st.session_state[f"payment_status_month_{i}"] for i in range(1, 7)},
}

prob = predict(inputs)
label, css_class, color, recommendation = risk_label(prob)
pct  = round(prob * 100, 1)

# Row 1 — Risk Score
col1, col2 = st.columns([1, 1.6])

with col1:
    st.markdown("### 🎯 Default Probability")
    # Plotly gauge
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=pct,
        number={"suffix": "%", "font": {"size": 42, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#555"},
            "bar":  {"color": color, "thickness": 0.28},
            "steps": [
                {"range": [0,  30], "color": "#eafaf1"},
                {"range": [30, 60], "color": "#fef9e7"},
                {"range": [60, 100],"color": "#fdedec"},
            ],
            "threshold": {
                "line": {"color": "#2C3E50", "width": 4},
                "thickness": 0.8, "value": pct
            },
        },
        title={"text": "Default Risk Score", "font": {"size": 16}},
    ))
    fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=10))
    st.plotly_chart(fig_gauge, use_container_width=True)

with col2:
    st.markdown("### 📊 Risk Assessment")
    st.markdown(f"""
<div class="risk-card {css_class}">
<h2 style="margin:0; color:{color};">{label}</h2>
<p style="margin:8px 0 0; font-size:1.05rem;">Default probability: <strong>{pct}%</strong></p>
<hr style="border-color:{color}33; margin:10px 0">
<p style="margin:0; font-size:0.95rem;"><strong>Recommendation:</strong><br>{recommendation}</p>
</div>
""", unsafe_allow_html=True)

    # Quick metrics
    eng = engineer(inputs.copy())
    m1, m2, m3 = st.columns(3)
    with m1:
        util_pct = round(eng["credit_utilization"] * 100, 1)
        st.metric("Credit Utilization", f"{util_pct}%",
                  delta="High" if util_pct > 70 else "Normal",
                  delta_color="inverse" if util_pct > 70 else "normal")
    with m2:
        st.metric("Months Delayed", int(eng["months_delayed"]),
                  delta="⚠️ Delinquent" if eng["months_delayed"] > 0 else "✓ Clean",
                  delta_color="inverse" if eng["months_delayed"] > 0 else "normal")
    with m3:
        ratio_pct = round(eng["payment_to_bill_ratio"] * 100, 1)
        st.metric("Pay/Bill Ratio", f"{ratio_pct}%",
                  delta="Good" if ratio_pct > 80 else "Low",
                  delta_color="normal" if ratio_pct > 80 else "inverse")

st.divider()

# Row 2 — Feature breakdown & Payment history
col3, col4 = st.columns([1.2, 1])

with col3:
    st.markdown("### 🔍 Key Risk Drivers")
    drivers = {
        "Payment Status (M1)":    min(max(inputs["payment_status_month_1"] / 8, 0), 1),
        "Credit Utilization":     min(eng["credit_utilization"], 1),
        "Months with Delay":      eng["months_delayed"] / 6,
        "Avg Bill / Limit":       min(eng["avg_bill_6m"] / (inputs["credit_limit_mad"] or 1), 1),
        "Pay-to-Bill Ratio (inv)":1 - min(eng["payment_to_bill_ratio"], 1),
    }
    bar_colors = ["#C0392B" if v > 0.6 else "#F39C12" if v > 0.3 else "#27AE60"
                  for v in drivers.values()]
    fig_drivers = go.Figure(go.Bar(
        x=list(drivers.values()),
        y=list(drivers.keys()),
        orientation="h",
        marker_color=bar_colors,
        text=[f"{v*100:.0f}%" for v in drivers.values()],
        textposition="outside",
    ))
    fig_drivers.update_layout(
        xaxis=dict(range=[0, 1.15], title="Risk Contribution"),
        height=260, margin=dict(l=10, r=40, t=10, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_drivers, use_container_width=True)

with col4:
    st.markdown("### 📈 Statement Balance Trend (6 months)")
    months = ["M6 (Oldest)", "M5", "M4", "M3", "M2", "M1 (Recent)"]
    balances = [inputs[f"bill_amount_month_{i}"] for i in range(6, 0, -1)]
    payments = [inputs[f"payment_amount_month_{i}"] for i in range(6, 0, -1)]

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=months, y=balances, name="Balance",
        line=dict(color="#C0392B", width=2.5),
        fill="tozeroy", fillcolor="rgba(192,57,43,0.08)"
    ))
    fig_trend.add_trace(go.Bar(
        x=months, y=payments, name="Payment",
        marker_color="rgba(39,174,96,0.75)"
    ))
    fig_trend.update_layout(
        height=270, margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", y=1.1),
        yaxis_title="MAD",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_trend, use_container_width=True)

st.divider()

# Row 3 — Approval framework
st.markdown("### 🏦 Branch Manager Decision Framework")
fw1, fw2, fw3 = st.columns(3)

with fw1:
    st.markdown("""
<div class="risk-card low-risk">
<h4 style="color:#27AE60; margin:0">🟢 Low Risk &lt; 30%</h4>
<ul style="margin:8px 0 0; padding-left:18px; font-size:0.9rem">
<li>Approve with standard terms</li>
<li>Standard credit limit</li>
<li>Regular monitoring (quarterly)</li>
<li>Eligible for limit increase after 12 months</li>
</ul>
</div>""", unsafe_allow_html=True)

with fw2:
    st.markdown("""
<div class="risk-card medium-risk">
<h4 style="color:#F39C12; margin:0">🟡 Medium Risk 30–60%</h4>
<ul style="margin:8px 0 0; padding-left:18px; font-size:0.9rem">
<li>Approve with reduced limit (–30%)</li>
<li>Monthly statement review</li>
<li>Proactive repayment plan offer</li>
<li>Flag for 6-month reassessment</li>
</ul>
</div>""", unsafe_allow_html=True)

with fw3:
    st.markdown("""
<div class="risk-card high-risk">
<h4 style="color:#C0392B; margin:0">🔴 High Risk &gt; 60%</h4>
<ul style="margin:8px 0 0; padding-left:18px; font-size:0.9rem">
<li>Decline new credit</li>
<li>Freeze limit increases</li>
<li>Escalate to recovery team</li>
<li>Offer structured repayment plan</li>
</ul>
</div>""", unsafe_allow_html=True)

# Footer
st.divider()
st.markdown(
    "<p style='text-align:center; color:#888; font-size:0.8rem;'>"
    "🇲🇦 Moroccan Banking Credit Risk Suite · XGBoost Model · "
    "SHAP-explainable · Aligned with Bank Al-Maghrib risk guidelines · "
    "For internal risk team use only"
    "</p>",
    unsafe_allow_html=True,
)
