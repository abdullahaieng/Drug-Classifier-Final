# Drug AI — Drug Classification App

Simple structure:

```
drug200.csv
app.py
models/          ← joblib files (after export)
requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
python app.py export
streamlit run app.py
```

**Local:** `python app.py export` ek baar (optional — models `models/` mein save).

**Streamlit Cloud:** Repo mein `drug200.csv` + `.python-version` (3.12) + pinned `scikit-learn==1.6.1` rakho. Agar `.pkl` missing hon to app memory mein train karti hai (sklearn 1.7+ par LogisticRegression OvR-wrapped hai).

Predict par **retrain nahi** — sirf saved/auto-built models use hoti hain.

## Notebook parity

Pipeline = notebook jaisa: label encode (Sex, BP, Cholesterol) → `StandardScaler` → `predict` → drug name inverse transform.

Test accuracy: LR **85%**, KNN **87.5%**.

## Features

- Light / Dark medical theme (sidebar toggle)
- Notebook graphs: countplot, histplot+kde, confusion heatmaps
- Predictions = saved models only (notebook pipeline)

## Note

Educational demo — medical advice nahi.
