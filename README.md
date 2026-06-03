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

`export` sirf ek baar chalao (models save). App startup par sirf saved `.pkl` load karti hai — **retrain nahi**.

## Notebook parity

Pipeline = notebook jaisa: label encode (Sex, BP, Cholesterol) → `StandardScaler` → `predict` → drug name inverse transform.

Test accuracy: LR **85%**, KNN **87.5%**.

## Features

- Light / Dark medical theme (sidebar toggle)
- Notebook graphs: countplot, histplot+kde, confusion heatmaps
- Predictions = saved models only (notebook pipeline)

## Note

Educational demo — medical advice nahi.
