# Customer Churn Prediction with Explainability

A machine learning app that predicts customer churn risk and explains *why*
using SHAP, deployed as an interactive Streamlit dashboard.

**Live demo:** _(add your Streamlit Cloud link here once deployed)_

---

## Project Structure

```
churn_project/
├── data/
│   └── telco_churn.csv        <- you'll download this (step 1)
├── models/                    <- trained model + SHAP plots get saved here
├── app/
│   └── app.py                 <- Streamlit dashboard
├── train_model.py             <- trains and saves the model
├── shap_analysis.py           <- generates SHAP explanation plots
├── requirements.txt
└── README.md
```

---

## Step 1 — Get the dataset

1. Go to Kaggle: search **"Telco Customer Churn"** (IBM sample dataset, ~7,000 rows).
   Direct link: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
2. Download `WA_Fn-UseC_-Telco-Customer-Churn.csv`
3. Rename it to `telco_churn.csv` and place it in `data/telco_churn.csv`

(You'll need a free Kaggle account to download.)

---

## Step 2 — Set up your environment

```bash
cd churn_project
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Step 3 — Train the model

```bash
python train_model.py
```

This will:
- Clean the data (fix missing `TotalCharges`, encode categoricals)
- Train Logistic Regression (baseline) and Random Forest (with a small grid of settings)
- Print precision, recall, F1, and ROC-AUC for each — **don't just look at accuracy**,
  churn datasets are imbalanced (~27% churn rate), so a model that always predicts
  "no churn" would look "accurate" while being useless
- Save the best model to `models/churn_model.pkl`

Expect Random Forest to win, typically landing around F1 ≈ 0.55–0.62 and
ROC-AUC ≈ 0.82–0.85 on this dataset — solid numbers to quote in your resume/README.

---

## Step 4 — Generate SHAP explanations

```bash
python shap_analysis.py
```

This creates two images in `models/`:
- **`shap_summary.png`** — which features matter most *overall* (usually: contract
  type, tenure, and monthly charges dominate)
- **`shap_waterfall_0.png`** — a specific customer's prediction broken down feature
  by feature

Use these two images in your README/LinkedIn post — they're the most convincing
part of the project because they show you understand *interpretability*, not just
"I called `.fit()`".

---

## Step 5 — Run the Streamlit app locally

```bash
streamlit run app/app.py
```

This opens a browser window where you can:
- Fill in a customer's details via form inputs
- Get a churn probability + prediction
- See a live SHAP waterfall plot explaining that specific prediction

Test it with a few different inputs — try a month-to-month, high-monthly-charge,
low-tenure customer (should show high risk) vs. a two-year contract, low-charge,
long-tenure customer (should show low risk). If the direction of these makes
sense, your model is behaving sanely.

---

## Step 6 — Push to GitHub

```bash
git init
git add .
git commit -m "Customer churn prediction with SHAP explainability"
```

Create a `.gitignore` so you don't commit the venv or raw data:

```
venv/
__pycache__/
data/telco_churn.csv
```

Then create a repo on GitHub and push:

```bash
git remote add origin https://github.com/<your-username>/churn-prediction-shap.git
git branch -M main
git push -u origin main
```

> Note: since you're excluding the raw CSV, mention in your README where to
> download the dataset (link above) so anyone cloning the repo can reproduce it.
> Alternatively, commit the CSV if it's small enough and licensing allows it.

---

## Step 7 — Deploy on Streamlit Community Cloud (free)

1. Go to https://share.streamlit.io and sign in with GitHub
2. Click **"New app"**
3. Select your repo, branch `main`, and set the main file path to `app/app.py`
4. Click **Deploy**

**Important:** Streamlit Cloud needs the trained model file (`models/churn_model.pkl`
and `models/feature_columns.json`) to exist in the repo, since it can't run
`train_model.py` for you. So:
- Either commit the `models/` folder (the `.pkl` files are usually small, a few MB)
- Or add a step in `app.py` that trains the model on first run if the file is
  missing (more advanced — commit the model for now, simplest path)

Once deployed, you'll get a public URL like:
`https://your-app-name.streamlit.app`

Add this link to your resume, GitHub README, and LinkedIn.

---

## Step 8 — Polish for your resume/portfolio

Update your resume project line from *"In Progress"* to include:
- The live demo link
- Your actual F1/ROC-AUC numbers (from Step 3's output)
- One sentence on the SHAP insight, e.g. *"Model identifies month-to-month
  contracts and low tenure as the strongest churn predictors, validated via SHAP."*

Example resume bullet:

> Built and deployed an explainable churn prediction model (Random Forest,
> ROC-AUC 0.84) with SHAP-based interpretability; created an interactive
> Streamlit dashboard allowing per-customer risk exploration — [live demo link]

---

## Troubleshooting

- **`FileNotFoundError` for the CSV**: make sure it's at `data/telco_churn.csv`
  exactly, and you're running commands from the `churn_project/` root folder.
- **SHAP waterfall plot errors**: SHAP's API has changed across versions;
  if `shap.plots._waterfall.waterfall_legacy` errors, try
  `shap.plots.waterfall(shap.Explanation(...))` instead — check your installed
  SHAP version with `pip show shap`.
- **Streamlit Cloud deployment fails**: check the app logs on the Streamlit
  Cloud dashboard — usually a missing package in `requirements.txt` or a
  missing model file.
