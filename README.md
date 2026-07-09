# Drug AI

A simple machine learning project built with Streamlit that predicts the recommended drug based on patient information from the Drug200 dataset.

## Project Structure

```text
Drug-AI/
│
├── app.py
├── drug200.csv
├── requirements.txt
├── .python-version
├── models/
│   ├── logistic_model.pkl
│   ├── knn_model.pkl
│   ├── scaler.pkl
│   ├── encoders.pkl
│   └── drug_encoder.pkl
└── README.md
```

## Installation

Clone the repository and install the required packages.

```bash
pip install -r requirements.txt
```

If you want to export the trained models locally, run:

```bash
python app.py export
```

Then start the app:

```bash
streamlit run app.py
```

---

## How it works

The application follows the same preprocessing steps used during model training.

* Encodes Sex, BP and Cholesterol
* Scales numerical values using StandardScaler
* Makes predictions with the selected model
* Converts the predicted label back to the original drug name

The app loads the saved models from the `models` folder. If they are not available, it trains them once in memory when the application starts.

Predictions never retrain the model.

---

## Models

Two models are included.

| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | 85%      |
| KNN                 | 87.5%    |

---

## Features

* Drug prediction
* Logistic Regression and KNN models
* Light and Dark mode
* Dataset visualizations
* Confusion matrix
* Streamlit interface
* Automatic model loading
* Streamlit Cloud support

---

## Streamlit Cloud

For deployment, keep these files in the repository.

* drug200.csv
* requirements.txt
* .python-version
* models folder (recommended)

If the model files are missing, the app creates them automatically when it starts.

---

## Requirements

* Python 3.12
* Streamlit
* scikit-learn 1.6.1
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Joblib

Install everything using:

```bash
pip install -r requirements.txt
```

---

## Note

This project was built for learning and demonstration purposes. It should not be used for real medical decisions.
