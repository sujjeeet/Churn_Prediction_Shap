"""
train_model.py
---------------
End-to-end training script for the Customer Churn Prediction project.

Steps:
1. Load the Telco Customer Churn dataset
2. Clean and preprocess
3. Train Logistic Regression (baseline) and Random Forest (main model)
4. Evaluate both with precision, recall, F1, ROC-AUC
5. Save the best model + feature columns for use in the Streamlit app

Run with:
    python train_model.py
"""

import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)

DATA_PATH = "data/telco_churn.csv"
MODEL_PATH = "models/churn_model.pkl"
SCALER_PATH = "models/scaler.pkl"
COLUMNS_PATH = "models/feature_columns.json"


def load_and_clean_data(path):
    """Load the Telco churn CSV and clean known issues."""
    df = pd.read_csv(path)

    # TotalCharges has some blank strings instead of numbers -- fix that
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # customerID isn't a useful feature
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Target column: Yes/No -> 1/0
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df


def preprocess(df):
    """One-hot encode categoricals, split features/target."""
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X = pd.get_dummies(X, drop_first=True)

    return X, y


def evaluate(model, X_test, y_test, name):
    """Print key classification metrics (accuracy alone is misleading here)."""
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    print(f"\n--- {name} ---")
    print(f"Precision: {precision_score(y_test, preds):.3f}")
    print(f"Recall:    {recall_score(y_test, preds):.3f}")
    print(f"F1 Score:  {f1_score(y_test, preds):.3f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, probs):.3f}")
    print(classification_report(y_test, preds))

    return f1_score(y_test, preds)


def main():
    print("Loading data...")
    df = load_and_clean_data(DATA_PATH)

    print(f"Dataset shape: {df.shape}")
    print(f"Churn rate: {df['Churn'].mean():.2%}")

    X, y = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale for Logistic Regression (Random Forest doesn't need this,
    # but we scale consistently and just use the tree model in the app)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Baseline: Logistic Regression ---
    log_reg = LogisticRegression(class_weight="balanced", max_iter=1000)
    log_reg.fit(X_train_scaled, y_train)
    f1_lr = evaluate(log_reg, X_test_scaled, y_test, "Logistic Regression (baseline)")

    # --- Main model: Random Forest ---
    best_rf = None
    best_f1 = -1
    for n_estimators in [100, 200]:
        for max_depth in [5, 10]:
            rf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                class_weight="balanced",
                random_state=42,
            )
            rf.fit(X_train, y_train)  # RF trained on unscaled data
            preds = rf.predict(X_test)
            f1 = f1_score(y_test, preds)
            print(f"RF (n_estimators={n_estimators}, max_depth={max_depth}) F1={f1:.3f}")
            if f1 > best_f1:
                best_f1 = f1
                best_rf = rf

    print("\nBest Random Forest configuration:")
    evaluate(best_rf, X_test, y_test, "Random Forest (best)")

    # Pick the overall winner between LR and RF (RF usually wins on this dataset)
    if best_f1 >= f1_lr:
        final_model = best_rf
        print("\n>>> Random Forest selected as final model.")
    else:
        final_model = log_reg
        print("\n>>> Logistic Regression selected as final model.")

    # Save everything the Streamlit app needs
    joblib.dump(final_model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    with open(COLUMNS_PATH, "w") as f:
        json.dump(list(X.columns), f)

    print(f"\nModel saved to {MODEL_PATH}")
    print(f"Feature columns saved to {COLUMNS_PATH}")


if __name__ == "__main__":
    main()
