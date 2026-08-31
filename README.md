# Customer Churn Prediction with Explainability

A machine learning app that predicts whether a customer is likely to churn, and explains why using SHAP. Built as part of my data science portfolio, deployed with Streamlit.

**Live demo:** [PASTE YOUR STREAMLIT CLOUD URL HERE]

---

## Results

I trained this on the IBM Telco Customer Churn dataset — about 7,000 customers, roughly 27% churn rate.

| Metric | Logistic Regression | Random Forest |
|---|---|---|
| Precision | 0.505 | 0.529 |
| Recall | 0.781 | 0.778 |
| F1 Score | 0.613 | 0.630 |
| ROC-AUC | 0.841 | 0.841 |

Random Forest ended up being the better model — similar ROC-AUC to Logistic Regression but a better F1 and precision, so it went with that.

According to SHAP, the biggest factors driving churn were **[fill in from shap_summary.png, e.g. contract type, tenure, and monthly charges]**.

---

## Project structure

```
churn_project/
├── data/
│   └── telco_churn.csv
├── models/
├── app/
│   └── app.py
├── train_model.py
├── shap_analysis.py
├── requirements.txt
└── README.md
```

---

## How to run this yourself

**1. Get the dataset**

Download the Telco Customer Churn dataset from Kaggle: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
Rename the file to `telco_churn.csv` and drop it in `data/`.

**2. Set up the environment**

```bash
cd churn_project
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**3. Train the model**

```bash
python train_model.py
```

Trains Logistic Regression as a baseline, then Random Forest with a few different settings, and keeps whichever scores better on F1. Accuracy isn't the metric to trust here since the dataset is imbalanced — a model that just predicts "no churn" every time would still look ~73% accurate while being useless.

**4. Generate the SHAP plots**

```bash
python shap_analysis.py
```

Outputs a global feature importance plot and a per-customer waterfall plot into `models/`.

**5. Run the app**

```bash
streamlit run app/app.py
```

(If Windows complains that `streamlit` isn't recognized, use `python -m streamlit run app/app.py` instead.)

Fill in a customer's details and it'll show you the churn probability along with a SHAP breakdown of what drove that specific prediction.

---

## Notes on things that tripped me up

- SHAP's API changed between versions — newer versions return a 3D numpy array for the tree explainer instead of a list, which broke the original waterfall plot code. Handled that with a version check in `shap_analysis.py` and `app.py`.
- On Windows, `streamlit run` sometimes isn't recognized directly depending on how it got installed — `python -m streamlit run` sidesteps that.
- `TotalCharges` in the raw dataset has some blank strings instead of actual missing values, so a plain `pd.to_numeric()` will throw unless you pass `errors="coerce"` first.

---

## Tech stack

Python, Pandas, Scikit-learn, SHAP, Streamlit, Matplotlib

---

## What I'd improve with more time

- Proper cross-validation instead of tuning against the test set directly
- Try XGBoost alongside Random Forest for comparison
- Expose more input fields in the app (currently a few features are hardcoded for simplicity)
