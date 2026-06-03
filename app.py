"""
Drug200 — Clinical Drug Classification (Streamlit)
Notebook-identical ML pipeline · Single-file production app.

Run GUI:     streamlit run app.py
Export once: python app.py export
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from io import StringIO
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

# =============================================================================
# Configuration
# =============================================================================
APP_VERSION = "2.1.0"  # sidebar — verify Streamlit Cloud pulled latest app.py

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "drug200.csv"
MODELS_DIR = ROOT / "models"
FEATURE_COLUMNS = ["Age", "Sex", "BP", "Cholesterol", "Na_to_K"]

MODEL_FILES = {
    "logistic": MODELS_DIR / "logistic_model.pkl",
    "knn": MODELS_DIR / "knn_model.pkl",
    "scaler": MODELS_DIR / "scaler.pkl",
    "encoders": MODELS_DIR / "encoders.pkl",
    "drug_encoder": MODELS_DIR / "drug_encoder.pkl",
}

RISK_LEVEL = {
    "DrugY": "Low",
    "drugA": "High",
    "drugB": "Moderate",
    "drugC": "Moderate",
    "drugX": "Low",
}

def build_logistic_regressor():
    """
    Notebook: LogisticRegression(solver='liblinear', random_state=42).
    sklearn>=1.7 (Streamlit Cloud) rejects liblinear for n_classes>=3 unless OvR-wrapped.
    Same one-vs-rest behaviour as older sklearn defaults; metrics unchanged on Drug200.
    """
    base = LogisticRegression(random_state=42, solver="liblinear", max_iter=1000)
    return OneVsRestClassifier(base)


RECOMMENDATION = {
    "DrugY": "Primary regimen Y is indicated. Schedule routine follow-up and monitor vitals.",
    "drugA": "Regimen A profile detected. Specialist consultation recommended before prescribing.",
    "drugB": "Regimen B may be suitable. Plan follow-up laboratory tests within two weeks.",
    "drugC": "Regimen C candidate — review contraindications and patient history carefully.",
    "drugX": "Alternative regimen X aligns with this profile. Confirm dosage with clinical guidelines.",
}


# =============================================================================
# Notebook preprocessing (exact order)
# =============================================================================
def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Duplicate removal + missing value handling (notebook)."""
    df = df.drop_duplicates().copy()
    numeric_cols = df.select_dtypes(include=["number"]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    cat_cols = df.select_dtypes(exclude="number").columns
    df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])
    return df


def notebook_train_pipeline(df_clean: pd.DataFrame) -> dict:
    """
    Replicate notebook: encode → split → scale → train.
    Used for export and for dynamic metrics (not at predict time).
    """
    X = df_clean.drop("Drug", axis=1)
    y = df_clean["Drug"]

    le_sex = LabelEncoder()
    le_bp = LabelEncoder()
    le_chol = LabelEncoder()
    le_drug = LabelEncoder()

    le_sex.fit(X["Sex"])
    le_bp.fit(X["BP"])
    le_chol.fit(X["Cholesterol"])
    y_enc = le_drug.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42
    )
    X_train = X_train.copy()
    X_test = X_test.copy()

    X_train["Sex"] = le_sex.transform(X_train["Sex"])
    X_test["Sex"] = le_sex.transform(X_test["Sex"])
    X_train["BP"] = le_bp.transform(X_train["BP"])
    X_test["BP"] = le_bp.transform(X_test["BP"])
    X_train["Cholesterol"] = le_chol.transform(X_train["Cholesterol"])
    X_test["Cholesterol"] = le_chol.transform(X_test["Cholesterol"])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logistic = build_logistic_regressor()
    knn = KNeighborsClassifier()
    logistic.fit(X_train_scaled, y_train)
    knn.fit(X_train_scaled, y_train)

    y_pred_lr = logistic.predict(X_test_scaled)
    y_pred_knn = knn.predict(X_test_scaled)

    def metrics(y_true, y_pred) -> dict:
        return {
            "accuracy": accuracy_score(y_true, y_pred) * 100,
            "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0) * 100,
            "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0) * 100,
            "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0) * 100,
        }

    lr_m = metrics(y_test, y_pred_lr)
    knn_m = metrics(y_test, y_pred_knn)

    return {
        "encoders": {"Sex": le_sex, "BP": le_bp, "Cholesterol": le_chol},
        "drug_encoder": le_drug,
        "scaler": scaler,
        "logistic": logistic,
        "knn": knn,
        "X_train_scaled": X_train_scaled,
        "train_shape": X_train.shape,
        "test_shape": X_test.shape,
        "scaled_head": X_train_scaled[:5],
        "drug_mapping": {i: c for i, c in enumerate(le_drug.classes_)},
        "y_test": y_test,
        "y_pred_lr": y_pred_lr,
        "y_pred_knn": y_pred_knn,
        "lr_metrics": lr_m,
        "knn_metrics": knn_m,
        "report_lr": classification_report(y_test, y_pred_lr, zero_division=0),
        "report_knn": classification_report(y_test, y_pred_knn, zero_division=0),
        "cm_lr": confusion_matrix(y_test, y_pred_lr),
        "cm_knn": confusion_matrix(y_test, y_pred_knn),
    }


def persist_bundle(bundle: dict) -> None:
    """Save trained artifacts; ignore disk errors (ephemeral cloud FS)."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    mapping = {
        "logistic": bundle["logistic"],
        "knn": bundle["knn"],
        "scaler": bundle["scaler"],
        "encoders": bundle["encoders"],
        "drug_encoder": bundle["drug_encoder"],
    }
    for key, obj in mapping.items():
        try:
            joblib.dump(obj, MODEL_FILES[key])
        except OSError:
            pass


def export_models() -> dict:
    """Train with notebook logic and persist joblib artifacts."""
    df = preprocess_dataframe(pd.read_csv(DATA_PATH))
    bundle = notebook_train_pipeline(df)
    persist_bundle(bundle)

    print(f"Artifacts saved to {MODELS_DIR}")
    print(f"Logistic Regression accuracy: {bundle['lr_metrics']['accuracy']:.2f}%")
    print(f"KNN accuracy: {bundle['knn_metrics']['accuracy']:.2f}%")
    return bundle


# =============================================================================
# Artifacts & inference
# =============================================================================
def _missing_artifacts() -> list[Path]:
    return [p for p in MODEL_FILES.values() if not p.exists()]


def _purge_artifacts() -> None:
    """Remove broken/incompatible pkl files so we can retrain."""
    for path in MODEL_FILES.values():
        path.unlink(missing_ok=True)
    load_artifacts.clear()
    _trained_bundle.clear()
    evaluation_bundle.clear()


def _load_from_disk() -> dict:
    return {
        "logistic": joblib.load(MODEL_FILES["logistic"]),
        "knn": joblib.load(MODEL_FILES["knn"]),
        "scaler": joblib.load(MODEL_FILES["scaler"]),
        "encoders": joblib.load(MODEL_FILES["encoders"]),
        "drug_encoder": joblib.load(MODEL_FILES["drug_encoder"]),
    }


@st.cache_resource
def _trained_bundle() -> dict:
    """Train once in memory when pkl files are missing (Streamlit Cloud safe)."""
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset missing: {DATA_PATH}\n"
            "Add drug200.csv next to app.py in your GitHub repo."
        )
    df = preprocess_dataframe(pd.read_csv(DATA_PATH))
    bundle = notebook_train_pipeline(df)
    persist_bundle(bundle)
    return {
        "logistic": bundle["logistic"],
        "knn": bundle["knn"],
        "scaler": bundle["scaler"],
        "encoders": bundle["encoders"],
        "drug_encoder": bundle["drug_encoder"],
    }


@st.cache_resource
def load_artifacts() -> dict:
    """Load saved models or train in memory (Streamlit Cloud safe)."""
    if not _missing_artifacts():
        try:
            return _load_from_disk()
        except Exception:
            _purge_artifacts()
    return _trained_bundle()


def _encoded_train_test(df: pd.DataFrame, art: dict) -> tuple:
    """Train/test split with saved encoders (no retrain)."""
    encoders = art["encoders"]
    drug_enc = art["drug_encoder"]
    X = df.drop("Drug", axis=1).copy()
    y = drug_enc.transform(df["Drug"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train = X_train.copy()
    X_test = X_test.copy()
    for col in ("Sex", "BP", "Cholesterol"):
        X_train[col] = encoders[col].transform(X_train[col])
        X_test[col] = encoders[col].transform(X_test[col])
    return X_train, X_test, y_train, y_test


@st.cache_data
def load_clean_data() -> pd.DataFrame:
    return preprocess_dataframe(pd.read_csv(DATA_PATH))


@st.cache_data
def evaluation_bundle(_cache_key: int = 1) -> dict:
    """Dynamic metrics from saved artifacts + notebook train/test split."""
    df_raw = pd.read_csv(DATA_PATH)
    df = load_clean_data()
    art = load_artifacts()
    scaler = art["scaler"]
    drug_enc = art["drug_encoder"]
    logistic = art["logistic"]
    knn = art["knn"]

    X_train, X_test, _, y_test = _encoded_train_test(df, art)
    X_test_scaled = scaler.transform(X_test)

    y_pred_lr = logistic.predict(X_test_scaled)
    y_pred_knn = knn.predict(X_test_scaled)

    def calc_metrics(y_true, y_pred) -> dict:
        return {
            "accuracy": accuracy_score(y_true, y_pred) * 100,
            "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0) * 100,
            "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0) * 100,
            "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0) * 100,
        }

    lr_m = calc_metrics(y_test, y_pred_lr)
    knn_m = calc_metrics(y_test, y_pred_knn)
    best = "Logistic Regression" if lr_m["accuracy"] >= knn_m["accuracy"] else "KNN"

    return {
        "df_raw": df_raw,
        "df_clean": df,
        "duplicates_removed": int(df_raw.duplicated().sum()),
        "train_shape": X_train.shape,
        "test_shape": X_test.shape,
        "scaled_head": scaler.transform(X_train)[:5],
        "drug_mapping": {i: c for i, c in enumerate(drug_enc.classes_)},
        "encoders": art["encoders"],
        "lr_metrics": lr_m,
        "knn_metrics": knn_m,
        "best_model": best,
        "report_lr": classification_report(y_test, y_pred_lr, zero_division=0),
        "report_knn": classification_report(y_test, y_pred_knn, zero_division=0),
        "cm_lr": confusion_matrix(y_test, y_pred_lr),
        "cm_knn": confusion_matrix(y_test, y_pred_knn),
        "class_labels": list(drug_enc.classes_),
    }


def preprocess_input(
    age: int,
    sex: str,
    bp: str,
    cholesterol: str,
    na_to_k: float,
    encoders: dict,
    scaler: StandardScaler,
) -> np.ndarray:
    """Single-row transform: label encode → scale (notebook inference path)."""
    row = pd.DataFrame([[age, sex, bp, cholesterol, na_to_k]], columns=FEATURE_COLUMNS)
    raw = {"Sex": sex, "BP": bp, "Cholesterol": cholesterol}
    for col in ("Sex", "BP", "Cholesterol"):
        le = encoders[col]
        val = raw[col]
        if val not in le.classes_:
            raise ValueError(f"Invalid {col}: {val}. Allowed: {list(le.classes_)}")
        row[col] = le.transform(row[col])
    return scaler.transform(row)


def predict_drug(model, X_scaled: np.ndarray, drug_encoder: LabelEncoder) -> dict:
    t0 = time.perf_counter()
    idx = int(model.predict(X_scaled)[0])
    proba_raw = model.predict_proba(X_scaled)
    proba = np.asarray(proba_raw[0], dtype=float).ravel()
    elapsed = (time.perf_counter() - t0) * 1000
    classes = drug_encoder.inverse_transform(np.arange(len(proba)))
    return {
        "drug": drug_encoder.inverse_transform([idx])[0],
        "confidence": float(proba.max()) * 100,
        "probabilities": dict(zip(classes, proba)),
        "time_ms": elapsed,
    }


# =============================================================================
# UI theme — balanced typography & medical palette
# =============================================================================
THEME = {
    "light": {
        "bg": "#eef2f6",
        "surface": "#ffffff",
        "sidebar": "#e8f4f2",
        "border": "#d1dde8",
        "text": "#1e293b",
        "muted": "#64748b",
        "accent": "#0f766e",
        "accent_soft": "#ccfbf1",
        "secondary": "#1d4ed8",
        "hero_grad": "linear-gradient(135deg, #0f766e 0%, #155e75 55%, #1e40af 100%)",
        "shadow": "0 8px 24px rgba(15, 118, 110, 0.12)",
    },
    "dark": {
        "bg": "#0f172a",
        "surface": "#1e293b",
        "sidebar": "#0c1222",
        "border": "#334155",
        "text": "#f1f5f9",
        "muted": "#94a3b8",
        "accent": "#2dd4bf",
        "accent_soft": "#134e4a",
        "secondary": "#60a5fa",
        "hero_grad": "linear-gradient(135deg, #115e59 0%, #0f766e 50%, #1e3a8a 100%)",
        "shadow": "0 8px 24px rgba(0, 0, 0, 0.35)",
    },
}


def inject_css(dark: bool) -> None:
    t = THEME["dark" if dark else "light"]

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        :root {{
            --bg: {t["bg"]};
            --surface: {t["surface"]};
            --border: {t["border"]};
            --text: {t["text"]};
            --muted: {t["muted"]};
            --accent: {t["accent"]};
            --accent-soft: {t["accent_soft"]};
            --secondary: {t["secondary"]};
            --fs-base: 15px;
            --fs-sm: 0.8125rem;
            --fs-md: 0.9375rem;
            --fs-lg: 1.125rem;
            --fs-xl: 1.375rem;
            --fs-2xl: 1.625rem;
        }}

        html, body, [class*="css"] {{
            font-family: 'Inter', system-ui, sans-serif !important;
            font-size: var(--fs-base);
            line-height: 1.5;
        }}

        .stApp {{ background: var(--bg) !important; }}

        [data-testid="stToolbar"], .stDeployButton, #MainMenu, footer {{
            display: none !important;
        }}

        .main .block-container {{
            max-width: 1100px;
            padding: 1.25rem 1.5rem 2rem;
        }}

        [data-testid="stSidebar"] {{
            background: {t["sidebar"]} !important;
            border-right: 1px solid var(--border);
        }}

        [data-testid="stSidebar"] .stMarkdown,
        [data-testid="stSidebar"] label {{
            color: var(--text) !important;
            font-size: var(--fs-md) !important;
        }}

        [data-testid="stSidebar"] h2 {{
            font-size: var(--fs-xl) !important;
            font-weight: 700 !important;
            margin-bottom: 0.15rem !important;
        }}

        [data-testid="stSidebar"] .stCaption {{
            font-size: var(--fs-sm) !important;
            color: var(--muted) !important;
        }}

        /* Headings — single scale */
        .main h1 {{ font-size: var(--fs-2xl) !important; font-weight: 700 !important; }}
        .main h2, [data-testid="stHeading"] h2 {{
            font-size: var(--fs-xl) !important;
            font-weight: 600 !important;
            color: var(--text) !important;
            margin-top: 1.25rem !important;
            margin-bottom: 0.65rem !important;
        }}
        .main h3 {{ font-size: var(--fs-lg) !important; font-weight: 600 !important; }}
        .main h4 {{ font-size: var(--fs-md) !important; font-weight: 600 !important; }}

        .main p, .main li, .main .stMarkdown, label {{
            color: var(--text) !important;
            font-size: var(--fs-md) !important;
        }}

        .stCaption {{ color: var(--muted) !important; font-size: var(--fs-sm) !important; }}

        /* Hero banner */
        .hero {{
            background: {t["hero_grad"]};
            padding: 1.25rem 1.5rem;
            border-radius: 14px;
            margin-bottom: 1.25rem;
            box-shadow: {t["shadow"]};
        }}
        .hero-title {{
            color: #ffffff !important;
            font-size: 1.5rem !important;
            font-weight: 700 !important;
            margin: 0 !important;
            line-height: 1.3 !important;
        }}
        .hero-sub {{
            color: rgba(255,255,255,0.92) !important;
            font-size: var(--fs-sm) !important;
            margin: 0.35rem 0 0 0 !important;
            font-weight: 400 !important;
        }}

        /* Prediction result cards */
        .pred-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1rem 1.1rem;
            margin-bottom: 0.5rem;
        }}
        .pred-card.lr {{ border-top: 3px solid var(--accent); }}
        .pred-card.knn {{ border-top: 3px solid var(--secondary); }}
        .pred-card.knn .pred-drug {{ color: var(--secondary) !important; }}
        .pred-model {{
            color: var(--muted) !important;
            font-size: var(--fs-sm) !important;
            font-weight: 600 !important;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin: 0 0 0.25rem 0 !important;
        }}
        .pred-drug {{
            color: var(--accent) !important;
            font-size: var(--fs-2xl) !important;
            font-weight: 700 !important;
            margin: 0 0 0.75rem 0 !important;
            line-height: 1.2 !important;
        }}
        .pred-rec {{
            color: var(--muted) !important;
            font-size: var(--fs-sm) !important;
            line-height: 1.45 !important;
            margin-top: 0.5rem !important;
        }}

        /* Metrics */
        div[data-testid="stMetric"] {{
            background: var(--surface);
            padding: 0.65rem 0.85rem;
            border-radius: 10px;
            border: 1px solid var(--border);
        }}
        [data-testid="stMetricLabel"] {{
            font-size: var(--fs-sm) !important;
            color: var(--muted) !important;
            font-weight: 500 !important;
        }}
        [data-testid="stMetricValue"] {{
            font-size: 1.2rem !important;
            font-weight: 700 !important;
            color: var(--accent) !important;
        }}

        /* Inputs & buttons */
        .stSlider label, .stSelectbox label {{
            font-size: var(--fs-sm) !important;
            font-weight: 500 !important;
            color: var(--muted) !important;
        }}

        div[data-testid="stVerticalBlockBorderWrapper"] {{
            border-color: var(--border) !important;
            border-radius: 12px !important;
            background: var(--surface) !important;
            padding: 0.25rem;
        }}

        .stButton button {{
            font-size: var(--fs-md) !important;
            border-radius: 10px !important;
        }}
        .stButton button[kind="primary"] {{
            background: var(--accent) !important;
            color: #fff !important;
            border: 0 !important;
            font-weight: 600 !important;
        }}
        .stButton button[kind="primary"]:hover {{
            filter: brightness(1.08);
        }}

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 6px;
        }}
        .stTabs [data-baseweb="tab"] {{
            font-size: var(--fs-sm) !important;
            font-weight: 600 !important;
            padding: 0.5rem 1rem !important;
        }}

        /* Dataframes & code */
        .stDataFrame, [data-testid="stDataFrame"] {{
            font-size: var(--fs-sm) !important;
        }}
        .stCode, pre {{
            font-size: 0.8rem !important;
        }}

        /* Alerts */
        .stAlert {{
            font-size: var(--fs-md) !important;
            border-radius: 10px !important;
        }}

        hr {{ border-color: var(--border) !important; margin: 1rem 0 !important; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_hero(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="hero">
          <p class="hero-title">{title}</p>
          <p class="hero-sub">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pred_card(model_name: str, pred: dict, css_class: str) -> None:
    risk = RISK_LEVEL.get(pred["drug"], "Unknown")
    rec = RECOMMENDATION.get(pred["drug"], "Consult clinical guidelines.")
    st.markdown(
        f"""
        <div class="pred-card {css_class}">
          <p class="pred-model">{model_name}</p>
          <p class="pred-drug">{pred["drug"]}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    m1, m2, m3 = st.columns(3)
    m1.metric("Confidence", f"{pred['confidence']:.1f}%")
    m2.metric("Risk", risk)
    m3.metric("Latency", f"{pred['time_ms']:.1f} ms")
    st.progress(min(pred["confidence"] / 100.0, 1.0))
    st.markdown(f'<p class="pred-rec">{rec}</p>', unsafe_allow_html=True)


def mpl_theme(dark: bool) -> None:
    t = THEME["dark" if dark else "light"]
    if dark:
        plt.style.use("dark_background")
        sns.set_theme(style="darkgrid", palette="Set2")
    else:
        sns.set_theme(style="whitegrid", palette="Set2")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.titlesize": 13,
            "axes.edgecolor": t["border"],
            "axes.labelcolor": t["text"],
            "text.color": t["text"],
        }
    )


def show_figure(fig: plt.Figure) -> None:
    st.pyplot(fig)
    plt.close(fig)


# =============================================================================
# Session state
# =============================================================================
def init_session() -> None:
    defaults = {
        "dark_mode": False,
        "page": "Prediction",
        "history": [],
        "last_prediction": None,
        "age": 40,
        "sex": "F",
        "bp": "HIGH",
        "cholesterol": "HIGH",
        "na_to_k": 15.0,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# =============================================================================
# Prediction page
# =============================================================================
def render_prediction_page(df: pd.DataFrame, artifacts: dict, dark: bool) -> None:
    page_hero(
        "Drug Recommendation System",
        "Real-time inference · Logistic Regression & KNN · Drug200 dataset",
    )

    age_lo, age_hi = int(df["Age"].min()), int(df["Age"].max())
    na_lo, na_hi = float(df["Na_to_K"].min()), float(df["Na_to_K"].max())

    with st.container(border=True):
        st.subheader("Patient parameters")
        c1, c2 = st.columns(2)
        age = c1.slider("Age", age_lo, age_hi, int(st.session_state.age))
        na_to_k = c2.slider("Na_to_K", na_lo, na_hi, float(st.session_state.na_to_k), 0.001, format="%.3f")
        sex = st.selectbox("Sex", ["F", "M"], index=["F", "M"].index(st.session_state.sex))
        bp = st.selectbox("Blood Pressure", ["HIGH", "NORMAL", "LOW"], index=["HIGH", "NORMAL", "LOW"].index(st.session_state.bp))
        chol = st.selectbox("Cholesterol", ["HIGH", "NORMAL"], index=["HIGH", "NORMAL"].index(st.session_state.cholesterol))
        st.session_state.update(age=age, sex=sex, bp=bp, cholesterol=chol, na_to_k=na_to_k)

        if st.button("Predict Drug", type="primary", use_container_width=True):
            try:
                X_scaled = preprocess_input(
                    age, sex, bp, chol, na_to_k,
                    artifacts["encoders"], artifacts["scaler"],
                )
                lr = predict_drug(artifacts["logistic"], X_scaled, artifacts["drug_encoder"])
                knn = predict_drug(artifacts["knn"], X_scaled, artifacts["drug_encoder"])
                st.session_state.last_prediction = {
                    "lr": lr, "knn": knn, "inputs": {
                        "age": age, "sex": sex, "bp": bp, "cholesterol": chol, "na_to_k": na_to_k,
                    }
                }
                st.session_state.history.insert(
                    0,
                    {
                        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Age": age,
                        "Sex": sex,
                        "BP": bp,
                        "Cholesterol": chol,
                        "Na_to_K": round(na_to_k, 3),
                        "LR_Drug": lr["drug"],
                        "LR_Confidence_%": round(lr["confidence"], 2),
                        "KNN_Drug": knn["drug"],
                        "KNN_Confidence_%": round(knn["confidence"], 2),
                    },
                )
            except ValueError as exc:
                st.error(str(exc))

    result = st.session_state.last_prediction
    if not result:
        st.info("Enter patient data and click **Predict Drug**.")
        return

    lr, knn = result["lr"], result["knn"]
    st.subheader("Prediction results")

    col1, col2 = st.columns(2)
    with col1:
        render_pred_card("Logistic Regression", lr, "lr")
    with col2:
        render_pred_card("KNN", knn, "knn")

    st.subheader("Prediction probability")
    mpl_theme(dark)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, title, pred in zip(axes, ["Logistic Regression", "KNN"], [lr, knn]):
        drugs = list(pred["probabilities"].keys())
        vals = [v * 100 for v in pred["probabilities"].values()]
        sns.barplot(x=drugs, y=vals, ax=ax, palette="Set2")
        ax.set_title(title)
        ax.set_ylabel("Probability %")
        ax.set_ylim(0, 100)
    fig.tight_layout()
    show_figure(fig)

    st.subheader("Prediction history")
    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download history CSV",
            hist_df.to_csv(index=False).encode("utf-8"),
            file_name="prediction_history.csv",
            mime="text/csv",
            use_container_width=True,
        )
        if st.button("Clear history"):
            st.session_state.history = []
            st.rerun()
    else:
        st.caption("No predictions recorded yet.")


# =============================================================================
# Notebook Lab page
# =============================================================================
def plot_eda(df: pd.DataFrame, encoders: dict, dark: bool) -> None:
    mpl_theme(dark)
    order = df["Drug"].value_counts().index

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.countplot(x="Drug", data=df, palette="Set2", order=order, ax=ax)
    ax.set_title("Drug Class Distribution")
    fig.tight_layout()
    show_figure(fig)

    for col, title in [("Sex", "Sex"), ("BP", "BP"), ("Cholesterol", "Cholesterol")]:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.countplot(x=col, data=df, ax=ax)
        ax.set_title(f"{title} Distribution")
        fig.tight_layout()
        show_figure(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(df["Age"], kde=True, ax=ax)
    ax.set_title("Age Distribution")
    fig.tight_layout()
    show_figure(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(df["Na_to_K"], kde=True, ax=ax)
    ax.set_title("Na_to_K Distribution")
    fig.tight_layout()
    show_figure(fig)

    tmp = df.copy()
    tmp["Sex"] = encoders["Sex"].transform(tmp["Sex"])
    tmp["BP"] = encoders["BP"].transform(tmp["BP"])
    tmp["Cholesterol"] = encoders["Cholesterol"].transform(tmp["Cholesterol"])
    corr = tmp.drop(columns=["Drug"]).corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, square=True)
    ax.set_title("Feature Correlation Heatmap")
    fig.tight_layout()
    show_figure(fig)


def plot_model_comparison(ev: dict, dark: bool) -> None:
    mpl_theme(dark)
    metrics = ["accuracy", "precision", "recall", "f1"]
    lr_vals = [ev["lr_metrics"][m] for m in metrics]
    knn_vals = [ev["knn_metrics"][m] for m in metrics]
    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width / 2, lr_vals, width, label="Logistic Regression", color="#0d9488")
    ax.bar(x + width / 2, knn_vals, width, label="KNN", color="#2563eb")
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.set_ylabel("Score (%)")
    ax.set_title("Model Comparison")
    ax.legend()
    ax.set_ylim(0, 100)
    fig.tight_layout()
    show_figure(fig)


def plot_confusion(cm: np.ndarray, title: str, dark: bool) -> None:
    mpl_theme(dark)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues" if not dark else "viridis", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    show_figure(fig)


def render_notebook_lab(ev: dict, dark: bool) -> None:
    page_hero(
        "Notebook Lab",
        "EDA · Training pipeline · Evaluation — mirrors DRUG_CLASSIFER.ipynb",
    )

    tab_data, tab_eda, tab_eval, tab_conc = st.tabs(
        ["Dataset", "EDA & Correlation", "Evaluation", "Conclusion"]
    )

    with tab_data:
        st.subheader("Dataset preview")
        st.dataframe(ev["df_raw"], use_container_width=True, height=220)
        c_h, c_t = st.columns(2)
        with c_h:
            st.subheader("Head")
            st.dataframe(ev["df_raw"].head(), use_container_width=True)
        with c_t:
            st.subheader("Tail")
            st.dataframe(ev["df_raw"].tail(), use_container_width=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Shape (rows)", ev["df_raw"].shape[0])
        c2.metric("Shape (cols)", ev["df_raw"].shape[1])
        c3.metric("Duplicates removed", ev["duplicates_removed"])
        buf = StringIO()
        ev["df_clean"].info(buf=buf)
        st.subheader("Info summary")
        st.code(buf.getvalue())
        st.subheader("Missing values (cleaned)")
        st.dataframe(ev["df_clean"].isnull().sum().to_frame("count"), use_container_width=True)
        st.subheader("Statistical summary")
        st.dataframe(ev["df_clean"].describe(include="all").transpose(), use_container_width=True)

        st.subheader("Label encoding mapping (Drug)")
        mapping_df = pd.DataFrame(
            [{"Code": k, "Drug": v} for k, v in ev["drug_mapping"].items()]
        )
        st.dataframe(mapping_df, use_container_width=True, hide_index=True)
        st.subheader("Scaled training sample")
        st.dataframe(
            pd.DataFrame(ev["scaled_head"], columns=FEATURE_COLUMNS),
            use_container_width=True,
        )

    with tab_eda:
        plot_eda(ev["df_clean"], ev["encoders"], dark)

    with tab_eval:
        st.subheader("Model training (notebook)")
        c1, c2 = st.columns(2)
        with c1:
            st.code(
                "OneVsRestClassifier(LogisticRegression(random_state=42, solver='liblinear'))"
            )
            st.caption("OvR wrapper required for sklearn>=1.7 + liblinear + 5 classes (notebook-equivalent).")
            st.metric("Train shape", str(ev["train_shape"]))
        with c2:
            st.code("KNeighborsClassifier()")
            st.metric("Test shape", str(ev["test_shape"]))

        st.success(f"Best model (by accuracy): **{ev['best_model']}**")
        plot_model_comparison(ev, dark)

        for label, prefix in [("Logistic Regression", "lr"), ("KNN", "knn")]:
            m = ev[f"{prefix}_metrics"]
            st.subheader(label)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{m['accuracy']:.2f}%")
            c2.metric("Precision", f"{m['precision']:.2f}%")
            c3.metric("Recall", f"{m['recall']:.2f}%")
            c4.metric("F1 Score", f"{m['f1']:.2f}%")
            st.caption("Classification report")
            st.code(ev[f"report_{prefix}"])
            plot_confusion(ev[f"cm_{prefix}"], f"{label} Confusion Matrix", dark)

    with tab_conc:
        st.subheader("Objective achieved")
        st.write(
            "The Drug200 dataset was successfully analyzed and machine learning models "
            "were used to predict suitable drugs using the same preprocessing and algorithms "
            "as the reference notebook."
        )
        st.subheader("Model performance")
        st.write(
            f"- **Logistic Regression** — Accuracy: {ev['lr_metrics']['accuracy']:.2f}%, "
            f"Precision: {ev['lr_metrics']['precision']:.2f}%, "
            f"Recall: {ev['lr_metrics']['recall']:.2f}%, "
            f"F1: {ev['lr_metrics']['f1']:.2f}%"
        )
        st.write(
            f"- **KNN** — Accuracy: {ev['knn_metrics']['accuracy']:.2f}%, "
            f"Precision: {ev['knn_metrics']['precision']:.2f}%, "
            f"Recall: {ev['knn_metrics']['recall']:.2f}%, "
            f"F1: {ev['knn_metrics']['f1']:.2f}%"
        )
        st.info(f"Best model: **{ev['best_model']}**")
        st.subheader("Future improvements")
        st.markdown(
            """
            - Collect more patient records to reduce class imbalance (DrugY dominance).
            - Try cross-validation and hyperparameter tuning for KNN (`n_neighbors`).
            - Add SHAP or LIME explainability for clinical trust.
            - Deploy with authentication and audit logs for hospital use.
            - Integrate real-time vitals from wearable devices.
            """
        )


# =============================================================================
# Main app
# =============================================================================
def run_app() -> None:
    try:
        st.set_page_config(
            page_title="DrugAI | Drug200",
            page_icon="💊",
            layout="wide",
            initial_sidebar_state="expanded",
        )
        init_session()
        inject_css(st.session_state.dark_mode)

        with st.spinner("Loading models…"):
            artifacts = load_artifacts()
        df = load_clean_data()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.info("Repo root mein `drug200.csv` add karo, phir Reboot app.")
        st.stop()
    except Exception as exc:
        st.error("App start nahi ho saki. Neeche error detail hai:")
        st.exception(exc)
        st.stop()

    with st.sidebar:
        st.markdown("## DrugAI")
        st.caption("Final Year Project · Drug Classification")
        st.session_state.dark_mode = st.toggle("Dark mode", value=st.session_state.dark_mode)
        inject_css(st.session_state.dark_mode)
        st.divider()
        st.session_state.page = st.radio(
            "Navigation",
            ["Prediction", "Notebook Lab"],
            index=0 if st.session_state.page == "Prediction" else 1,
        )
        st.divider()
        if st.session_state.page == "Notebook Lab":
            with st.spinner("Computing metrics..."):
                ev = evaluation_bundle()
            st.metric("Best model", ev["best_model"])
            st.metric("LR Accuracy", f"{ev['lr_metrics']['accuracy']:.2f}%")
            st.metric("KNN Accuracy", f"{ev['knn_metrics']['accuracy']:.2f}%")
        st.caption("Saved models on disk · Predict uses export only")
        st.caption(f"Build {APP_VERSION}")

    if st.session_state.page == "Prediction":
        render_prediction_page(df, artifacts, st.session_state.dark_mode)
    else:
        render_notebook_lab(evaluation_bundle(), st.session_state.dark_mode)

    st.divider()
    st.caption("Educational demonstration only — not for clinical diagnosis without validation.")


if "export" in sys.argv:
    export_models()
else:
    # Streamlit Cloud always runs this branch (not only __main__).
    run_app()
