"""
shap_analysis.py
------------------
Generates SHAP explainability plots for the trained churn model.
Run this AFTER train_model.py.

Outputs:
- models/shap_summary.png   -> global feature importance
- models/shap_waterfall_0.png -> explanation for one specific customer

Run with:
    python shap_analysis.py
"""

import pandas as pd
import numpy as np
import joblib
import json
import shap
import matplotlib.pyplot as plt

DATA_PATH = "data/telco_churn.csv"
MODEL_PATH = "models/churn_model.pkl"
COLUMNS_PATH = "models/feature_columns.json"


def load_and_clean_data(path):
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    return df


def main():
    model = joblib.load(MODEL_PATH)
    with open(COLUMNS_PATH) as f:
        feature_columns = json.load(f)

    df = load_and_clean_data(DATA_PATH)
    X = df.drop(columns=["Churn"])
    X = pd.get_dummies(X, drop_first=True)

    # Align columns with what the model was trained on
    X = X.reindex(columns=feature_columns, fill_value=0)

    # Use a sample for speed if the dataset is large
    X_sample = X.sample(min(500, len(X)), random_state=42)

    print("Building SHAP explainer (TreeExplainer works for Random Forest)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

        # SHAP's output format changed across versions:
    # - Older versions: shap_values is a list [array_for_class0, array_for_class1]
    # - Newer versions: shap_values is a single 3D array (n_samples, n_features, n_classes)
    # This handles both so the script works regardless of installed SHAP version.
    if isinstance(shap_values, list):
        shap_values_churn = shap_values[1]
        expected_value = explainer.expected_value[1] if hasattr(explainer.expected_value, "__len__") else explainer.expected_value
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_values_churn = shap_values[:, :, 1]
        expected_value = explainer.expected_value[1] if hasattr(explainer.expected_value, "__len__") else explainer.expected_value
    else:
        shap_values_churn = shap_values
        expected_value = explainer.expected_value[0] if hasattr(explainer.expected_value, "__len__") else explainer.expected_value

    print("Generating global summary plot...")
    plt.figure()
    shap.summary_plot(shap_values_churn, X_sample, show=False)
    plt.tight_layout()
    plt.savefig("models/shap_summary.png", dpi=150)
    plt.close()

    print("Generating per-customer waterfall plot (first customer in sample)...")
    explanation = shap.Explanation(
        values=shap_values_churn[0],
        base_values=expected_value,
        data=X_sample.iloc[0].values,
        feature_names=list(X_sample.columns),
    )
    plt.figure()
    shap.plots.waterfall(explanation, show=False)
    plt.tight_layout()
    plt.savefig("models/shap_waterfall_0.png", dpi=150)
    plt.close()

    print("Done. Check models/shap_summary.png and models/shap_waterfall_0.png")


if __name__ == "__main__":
    main()
