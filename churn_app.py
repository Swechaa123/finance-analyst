"""
Bank Customer Churn — Predictive Risk Dashboard
Run locally with:  streamlit run churn_app.py
Only requires churn_features.csv in the same folder — the model trains itself
on first launch (cached, so it only happens once per session).
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance

st.set_page_config(page_title="Bank Churn Risk Dashboard", layout="wide")

# ---------- Load data & train model (cached — runs once) ----------
@st.cache_data
def load_data():
    return pd.read_csv("churn_features.csv")

@st.cache_resource
def train_model(data):
    feature_cols = [c for c in data.columns if c != "Exited"]
    X = data[feature_cols]
    y = data["Exited"]
    model = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
    model.fit(X, y)
    return model, feature_cols

data = load_data()
with st.spinner("Training model (first run only, takes a few seconds)..."):
    model, feature_cols = train_model(data)

st.title("🏦 Bank Customer Churn — Predictive Risk Dashboard")
st.caption("Predictive Modeling and Risk Scoring for Bank Customer Churn — Unified Mentor / European Central Bank dataset")

tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Risk Calculator", "📊 Portfolio Overview", "🔍 Feature Importance", "🧪 What-If Simulator"
])

# ============================================================
# TAB 1: Individual Risk Calculator
# ============================================================
with tab1:
    st.subheader("Customer Churn Risk Calculator")
    col1, col2, col3 = st.columns(3)

    with col1:
        credit_score = st.slider("Credit Score", 300, 850, 650)
        age = st.slider("Age", 18, 92, 40)
        tenure = st.slider("Tenure (years with bank)", 0, 10, 5)

    with col2:
        balance = st.number_input("Account Balance (€)", 0.0, 250000.0, 75000.0, step=1000.0)
        salary = st.number_input("Estimated Salary (€)", 0.0, 200000.0, 100000.0, step=1000.0)
        num_products = st.selectbox("Number of Products", [1, 2, 3, 4], index=1)

    with col3:
        has_card = st.radio("Has Credit Card?", ["Yes", "No"], horizontal=True)
        is_active = st.radio("Active Member?", ["Yes", "No"], horizontal=True)
        geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
        gender = st.selectbox("Gender", ["Female", "Male"])

    threshold = st.slider("Churn flag threshold", 0.05, 0.95, 0.5, 0.05,
                           help="Lower threshold catches more churners (higher recall) at the cost of more false alarms.")

    def build_input_row():
        row = {
            'CreditScore': credit_score, 'Age': age, 'Tenure': tenure, 'Balance': balance,
            'NumOfProducts': num_products, 'HasCrCard': 1 if has_card == "Yes" else 0,
            'IsActiveMember': 1 if is_active == "Yes" else 0, 'EstimatedSalary': salary,
        }
        row['BalanceSalaryRatio'] = row['Balance'] / (row['EstimatedSalary'] + 1)
        row['ProductDensity'] = row['Balance'] / (row['NumOfProducts'] + 1)
        row['EngagementProductInteraction'] = row['IsActiveMember'] * row['NumOfProducts']
        row['AgeTenureInteraction'] = row['Age'] * row['Tenure']
        row['ZeroBalance'] = 1 if row['Balance'] == 0 else 0
        row['IsSenior'] = 1 if row['Age'] >= 60 else 0
        row['Geography_Germany'] = 1 if geography == "Germany" else 0
        row['Geography_Spain'] = 1 if geography == "Spain" else 0
        row['Gender_Male'] = 1 if gender == "Male" else 0
        df_row = pd.DataFrame([row])[feature_cols]
        return df_row

    if st.button("Calculate Churn Risk", type="primary"):
        X_input = build_input_row()
        proba = model.predict_proba(X_input)[0, 1]
        flag = "🔴 HIGH RISK — Likely to churn" if proba >= threshold else "🟢 LOW RISK — Likely to stay"

        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Churn Probability", f"{proba:.1%}")
            st.markdown(f"### {flag}")
        with c2:
            fig, ax = plt.subplots(figsize=(6, 1.2))
            ax.barh([0], [proba], color='#EF4444' if proba >= threshold else '#22C55E')
            ax.axvline(threshold, color='black', linestyle='--', label=f'Threshold ({threshold:.2f})')
            ax.set_xlim(0, 1)
            ax.set_yticks([])
            ax.legend(loc='upper right')
            ax.set_xlabel('Churn Probability')
            st.pyplot(fig)

# ============================================================
# TAB 2: Portfolio Overview
# ============================================================
with tab2:
    st.subheader("Portfolio-Wide Churn Probability Distribution")
    X_all = data[feature_cols]
    probs_all = model.predict_proba(X_all)[:, 1]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(probs_all[data['Exited'] == 0], bins=40, alpha=0.6, label='Retained (actual)', color='#3B82F6')
    ax.hist(probs_all[data['Exited'] == 1], bins=40, alpha=0.6, label='Churned (actual)', color='#EF4444')
    ax.set_xlabel("Predicted Churn Probability")
    ax.set_ylabel("Customer Count")
    ax.legend()
    st.pyplot(fig)

    st.markdown("#### Risk Segments")
    seg = pd.cut(probs_all, bins=[0, 0.3, 0.6, 1.0], labels=["Low", "Medium", "High"])
    seg_counts = seg.value_counts().reindex(["Low", "Medium", "High"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Low Risk (<30%)", int(seg_counts["Low"]))
    c2.metric("Medium Risk (30–60%)", int(seg_counts["Medium"]))
    c3.metric("High Risk (>60%)", int(seg_counts["High"]))

# ============================================================
# TAB 3: Feature Importance
# ============================================================
with tab3:
    st.subheader("What Drives Churn? (Permutation Importance)")
    X_all = data[feature_cols]
    y_all = data['Exited']
    with st.spinner("Computing permutation importance..."):
        perm = permutation_importance(model, X_all, y_all, n_repeats=8, random_state=42, scoring='roc_auc', n_jobs=-1)
    imp = pd.Series(perm.importances_mean, index=feature_cols).sort_values(ascending=False).head(12)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(imp.index[::-1], imp.values[::-1], color='#2563EB')
    ax.set_xlabel("Mean decrease in ROC-AUC when shuffled")
    st.pyplot(fig)
    st.caption("Higher bars = the feature matters more for distinguishing churners from retained customers.")

# ============================================================
# TAB 4: What-If Simulator
# ============================================================
with tab4:
    st.subheader("What-If Scenario Simulator")
    st.write("Pick a base customer profile from the dataset, then adjust engagement/product values to see how churn probability responds.")

    idx = st.number_input("Customer row index (0–9999)", 0, len(data) - 1, 0)
    base = data.iloc[idx].copy()

    st.write(f"Base profile — Age: {int(base['Age'])}, Products: {int(base['NumOfProducts'])}, "
             f"Active: {'Yes' if base['IsActiveMember'] == 1 else 'No'}, Balance: €{base['Balance']:,.0f}")

    sim_products = st.slider("Simulated NumOfProducts", 1, 4, int(base['NumOfProducts']))
    sim_active = st.radio("Simulated Active Member", ["Yes", "No"],
                           index=0 if base['IsActiveMember'] == 1 else 1, horizontal=True)
    sim_balance = st.slider("Simulated Balance (€)", 0.0, 250000.0, float(base['Balance']))

    sim = base.copy()
    sim['NumOfProducts'] = sim_products
    sim['IsActiveMember'] = 1 if sim_active == "Yes" else 0
    sim['Balance'] = sim_balance
    sim['BalanceSalaryRatio'] = sim['Balance'] / (sim['EstimatedSalary'] + 1)
    sim['ProductDensity'] = sim['Balance'] / (sim['NumOfProducts'] + 1)
    sim['EngagementProductInteraction'] = sim['IsActiveMember'] * sim['NumOfProducts']
    sim['ZeroBalance'] = 1 if sim['Balance'] == 0 else 0

    base_proba = model.predict_proba(pd.DataFrame([base[feature_cols]]))[0, 1]
    sim_proba = model.predict_proba(pd.DataFrame([sim[feature_cols]]))[0, 1]

    c1, c2 = st.columns(2)
    c1.metric("Original Churn Probability", f"{base_proba:.1%}")
    c2.metric("Simulated Churn Probability", f"{sim_proba:.1%}", delta=f"{(sim_proba - base_proba):+.1%}")
