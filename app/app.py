"""
app.py
-------
Streamlit dashboard for the Customer Churn Prediction project.

Run locally with:
    streamlit run app/app.py
"""

import streamlit as st
import numpy as np 
import pandas as pd
import joblib
import json
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

MODEL_PATH = "models/churn_model.pkl"
COLUMNS_PATH = "models/feature_columns.json"


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    with open(COLUMNS_PATH) as f:
        feature_columns = json.load(f)
    return model, feature_columns


model, feature_columns = load_model()

st.title("📊 Customer Churn Prediction with Explainability")
st.write(
    "Enter a customer's details below to predict their churn risk, "
    "and see exactly which factors drove the prediction using SHAP."
)

# --- Input form ---
with st.form("customer_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 70.0)
        total_charges = st.number_input("Total Charges ($)", 0.0, 9000.0, 800.0)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

    with col2:
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        )
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])

    with col3:
        partner = st.selectbox("Has Partner", ["No", "Yes"])
        dependents = st.selectbox("Has Dependents", ["No", "Yes"])
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])

    submitted = st.form_submit_button("Predict Churn Risk")

if submitted:
    # Build a single-row dataframe matching the raw feature format
    raw_input = {
        "gender": gender,
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": internet_service,
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": tech_support,
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    input_df = pd.DataFrame([raw_input])
    input_encoded = pd.get_dummies(input_df)

    # Align with training columns (fill missing dummy columns with 0)
    input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)

    # --- Prediction ---
    proba = model.predict_proba(input_encoded)[0][1]
    prediction = "Likely to Churn" if proba >= 0.5 else "Likely to Stay"

    st.subheader("Prediction Result")
    c1, c2 = st.columns(2)
    c1.metric("Churn Probability", f"{proba:.1%}")
    c2.metric("Prediction", prediction)

    if proba >= 0.5:
        st.warning("⚠️ This customer is at risk of churning.")
    else:
        st.success("✅ This customer is likely to stay.")

    # --- SHAP explanation for this specific prediction ---
    st.subheader("Why this prediction? (SHAP explanation)")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_encoded)
    expected_value = explainer.expected_value

    # Same version-handling as shap_analysis.py: newer SHAP returns a single
    # 3D array (n_samples, n_features, n_classes) instead of a list.
    if isinstance(shap_values, list):
        shap_values_churn = shap_values[1][0]
        ev = expected_value[1] if hasattr(expected_value, "__len__") else expected_value
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_values_churn = shap_values[0, :, 1]
        ev = expected_value[1] if hasattr(expected_value, "__len__") else expected_value
    else:
        shap_values_churn = shap_values[0]
        ev = expected_value[0] if hasattr(expected_value, "__len__") else expected_value

    explanation = shap.Explanation(
        values=shap_values_churn,
        base_values=ev,
        data=input_encoded.iloc[0].values,
        feature_names=list(input_encoded.columns),
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.plots.waterfall(explanation, show=False)
    st.pyplot(fig)

    st.caption(
        "Bars pushing right (red) increase churn risk; bars pushing left (blue) decrease it. "
        "The width of each bar shows how much that feature mattered for this specific customer."
    )

st.divider()
st.caption("Built with Scikit-learn, SHAP, and Streamlit | Telco Customer Churn dataset")
