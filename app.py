import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import shap
from xgboost import XGBClassifier

# ── Load model artifacts ──
@st.cache_resource
def load_artifacts():
    with open('models/xgb_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('models/feature_names.pkl', 'rb') as f:
        features = pickle.load(f)
    return model, scaler, features

model, scaler, feature_names = load_artifacts()

# ── Page config ──
st.set_page_config(page_title="Churn Predictor", page_icon="📊", layout="wide")
st.title("📊 Customer Churn Predictor")
st.caption("XGBoost model trained on IBM Telco dataset — 80% AUC")

# ── Tabs ──
tab1, tab2, tab3 = st.tabs(["🔮 Live Prediction", "📈 Model Performance", "🔍 SHAP Explainability"])

# ── Tab 1: Live Prediction ──
with tab1:
    st.subheader("Enter customer details")
    col1, col2, col3 = st.columns(3)

    with col1:
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = st.slider("Monthly Charges ($)", 18, 120, 65)
        total_charges = st.number_input("Total Charges ($)", 0.0, 9000.0, float(tenure * monthly_charges))

    with col2:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])

    with col3:
        gender = st.selectbox("Gender", ["Male", "Female"])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

    # Build input dataframe
    input_dict = {
        'gender': 1 if gender == 'Male' else 0,
        'SeniorCitizen': 0,
        'Partner': 1 if partner == 'Yes' else 0,
        'Dependents': 1 if dependents == 'Yes' else 0,
        'tenure': tenure,
        'PhoneService': 1 if phone_service == 'Yes' else 0,
        'PaperlessBilling': 1 if paperless_billing == 'Yes' else 0,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
    }

    # One-hot encode categoricals to match training features
    for feat in feature_names:
        if feat not in input_dict:
            input_dict[feat] = 0

    # Set the selected categories
    contract_map = {
        'Month-to-month': 'Contract_One year',
        'One year': 'Contract_One year',
        'Two year': 'Contract_Two year'
    }
    internet_map = {
        'Fiber optic': 'InternetService_Fiber optic',
        'No': 'InternetService_No'
    }
    payment_map = {
        'Credit card (automatic)': 'PaymentMethod_Credit card (automatic)',
        'Electronic check': 'PaymentMethod_Electronic check',
        'Mailed check': 'PaymentMethod_Mailed check'
    }

    for k, v in [
        (contract_map.get(contract), 1),
        (internet_map.get(internet_service), 1),
        (payment_map.get(payment_method), 1)
    ]:
        if k and k in input_dict:
            input_dict[k] = v

    input_df = pd.DataFrame([input_dict])[feature_names]

    # Scale numerical
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    input_df[num_cols] = scaler.transform(input_df[num_cols])

    # Predict
    if st.button("Predict Churn", type="primary"):
        prob = model.predict_proba(input_df)[0][1]
        pred = "Will Churn" if prob > 0.5 else "Will Not Churn"
        color = "red" if prob > 0.5 else "green"

        st.markdown("---")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Churn Probability", f"{prob:.1%}")
        with col_b:
            st.markdown(f"### :{color}[{pred}]")

        # Risk level
        if prob > 0.7:
            st.error("🔴 High risk — immediate retention action recommended")
        elif prob > 0.4:
            st.warning("🟡 Medium risk — monitor this customer")
        else:
            st.success("🟢 Low risk — customer likely to stay")

        # SHAP for this prediction
        st.markdown("#### Why this prediction?")
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(input_df)
        top_features = pd.Series(
            np.abs(shap_vals[0]),
            index=feature_names
        ).nlargest(5)
        st.bar_chart(top_features)

# ── Tab 2: Model Performance ──
with tab2:
    st.subheader("Model comparison — Logistic Regression vs Random Forest vs XGBoost")
    col1, col2 = st.columns(2)
    with col1:
        st.image("data/model_comparison.png", caption="AUC, F1, Precision, Recall")
    with col2:
        st.image("data/roc_curves.png", caption="ROC Curves")

    st.markdown("""
    **Key results:**
    - XGBoost achieves highest AUC (~0.84) and F1 score
    - Logistic Regression is competitive and more interpretable
    - Random Forest balances precision and recall well
    - Dataset is imbalanced (26% churn) — F1 is more meaningful than accuracy
    """)

# ── Tab 3: SHAP ──
with tab3:
    st.subheader("SHAP Feature Importance")
    col1, col2 = st.columns(2)
    with col1:
        st.image("data/shap_importance.png", caption="Global feature importance")
    with col2:
        st.image("data/shap_beeswarm.png", caption="Feature impact direction")

    st.markdown("""
    **Top churn drivers:**
    - **Tenure** — longer tenure = lower churn probability
    - **MonthlyCharges** — higher charges = higher churn risk
    - **Contract type** — month-to-month contracts churn most
    - **InternetService** — fiber optic customers churn more than DSL
    - **TotalCharges** — correlated with tenure
    """)
