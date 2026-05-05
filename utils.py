"""
utils.py — Shared helpers for the Diabetes Risk XAI application.

Provides model loading, preprocessing, risk tiering, and Plotly gauge chart.
"""

from __future__ import annotations

import os
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import joblib

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
MPL_CACHE_DIR = Path(tempfile.gettempdir()) / "diabetes_xai_matplotlib"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

# Raw columns from the PIMA dataset (no Outcome)
RAW_FEATURE_COLS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

# Columns where 0 is biologically impossible
ZERO_IMPUTE_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

# Engineered feature names
ENGINEERED_FEATURES = ["GlucoseBMI", "AgeRisk", "InsulinResistance"]

# All final features (raw + engineered)
ALL_FEATURES = RAW_FEATURE_COLS + ENGINEERED_FEATURES

# Risk tier boundaries
LOW_RISK_THRESHOLD = 0.35
HIGH_RISK_THRESHOLD = 0.65

# Model file paths
ENSEMBLE_MODEL_PATH = MODELS_DIR / "ensemble_model.pkl"
RF_MODEL_PATH = MODELS_DIR / "rf_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.pkl"
METRICS_PATH = MODELS_DIR / "metrics.json"

# ─────────────────────────────────────────────
# Model Loading
# ─────────────────────────────────────────────

def load_models():
    """
    Load all trained model artefacts from disk.

    Returns
    -------
    tuple
        (ensemble_model, rf_model, scaler, feature_names)

    Raises
    ------
    FileNotFoundError
        If any required .pkl file is missing.
    """
    missing = []
    for p in [ENSEMBLE_MODEL_PATH, RF_MODEL_PATH, SCALER_PATH, FEATURE_NAMES_PATH]:
        if not p.exists():
            missing.append(str(p))
    if missing:
        raise FileNotFoundError(
            "Missing model files:\n" + "\n".join(missing)
            + "\n\nPlease run:  python model.py"
        )

    ensemble = joblib.load(ENSEMBLE_MODEL_PATH)
    rf = joblib.load(RF_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(FEATURE_NAMES_PATH, "rb") as f:
        feature_names = pickle.load(f)

    return ensemble, rf, scaler, feature_names


# ─────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────

def preprocess(df: pd.DataFrame, scaler=None) -> np.ndarray:
    """
    Apply imputation, feature engineering, and scaling to a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input data containing RAW_FEATURE_COLS columns.
    scaler : sklearn StandardScaler, optional
        If provided, uses this scaler for transform. Otherwise returns
        unscaled engineered features (useful during training fit).

    Returns
    -------
    np.ndarray
        Scaled feature array ready for model inference.
    """
    df = df.copy()

    # Replace biologically impossible zeros with NaN
    for col in ZERO_IMPUTE_COLS:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)

    # Median imputation
    for col in ZERO_IMPUTE_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Feature engineering
    df["GlucoseBMI"] = df["Glucose"] * df["BMI"] / 100.0
    df["AgeRisk"] = df["Age"] * df["DiabetesPedigreeFunction"]
    df["InsulinResistance"] = df["Insulin"] / (df["BMI"] + 1.0)

    # Select final feature columns
    X = df[ALL_FEATURES]

    if scaler is not None:
        X = scaler.transform(X)
    else:
        X = X.values

    return X


def preprocess_with_median(df: pd.DataFrame, medians: dict, scaler) -> np.ndarray:
    """
    Apply imputation using pre-computed training medians, then scale.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input DataFrame.
    medians : dict
        Column → median value from training set.
    scaler : sklearn StandardScaler
        Fitted scaler.

    Returns
    -------
    np.ndarray
        Scaled feature array.
    """
    df = df.copy()

    # Replace zeros with NaN
    for col in ZERO_IMPUTE_COLS:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)

    # Fill NaN with training medians
    for col, median_val in medians.items():
        if col in df.columns:
            df[col] = df[col].fillna(median_val)

    # Feature engineering
    df["GlucoseBMI"] = df["Glucose"] * df["BMI"] / 100.0
    df["AgeRisk"] = df["Age"] * df["DiabetesPedigreeFunction"]
    df["InsulinResistance"] = df["Insulin"] / (df["BMI"] + 1.0)

    X = df[ALL_FEATURES]
    X = scaler.transform(X)
    return X


# ─────────────────────────────────────────────
# Risk Tiering
# ─────────────────────────────────────────────

def get_risk_tier(prob: float) -> tuple[str, str]:
    """
    Convert a probability to a risk tier label and display colour.

    Parameters
    ----------
    prob : float
        Predicted probability of diabetes (0–1).

    Returns
    -------
    tuple[str, str]
        (risk_label, hex_colour)
    """
    if prob < LOW_RISK_THRESHOLD:
        return "Low Risk", "#22c55e"          # green
    elif prob <= HIGH_RISK_THRESHOLD:
        return "Borderline Risk", "#f97316"   # orange
    else:
        return "High Risk", "#ef4444"         # red


# ─────────────────────────────────────────────
# Plotly Gauge Chart
# ─────────────────────────────────────────────

def plot_gauge(prob: float) -> go.Figure:
    """
    Create a Plotly gauge chart showing diabetes risk probability.

    The gauge has three coloured zones:
      Green  (0–35%)  → Low Risk
      Orange (35–65%) → Borderline Risk
      Red    (65–100%) → High Risk

    Parameters
    ----------
    prob : float
        Predicted probability of diabetes (0–1).

    Returns
    -------
    plotly.graph_objects.Figure
        Configured gauge figure.
    """
    risk_label, _ = get_risk_tier(prob)
    pct = prob * 100

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=pct,
        number={"suffix": "%", "font": {"size": 36, "color": "#f1f5f9"}},
        delta={
            "reference": 50,
            "increasing": {"color": "#ef4444"},
            "decreasing": {"color": "#22c55e"},
            "font": {"size": 14},
        },
        title={
            "text": f"<b>Diabetes Risk</b><br><span style='font-size:14px;color:#94a3b8'>{risk_label}</span>",
            "font": {"size": 18, "color": "#f1f5f9"},
        },
        gauge={
            "axis": {
                "range": [0, 100],
                "tickwidth": 1,
                "tickcolor": "#64748b",
                "tickfont": {"color": "#94a3b8", "size": 11},
            },
            "bar": {"color": "#818cf8", "thickness": 0.3},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 35], "color": "rgba(34,197,94,0.25)"},
                {"range": [35, 65], "color": "rgba(249,115,22,0.25)"},
                {"range": [65, 100], "color": "rgba(239,68,68,0.25)"},
            ],
            "threshold": {
                "line": {"color": "#f1f5f9", "width": 3},
                "thickness": 0.85,
                "value": pct,
            },
        },
    ))

    fig.update_layout(
        height=280,
        margin=dict(t=60, b=0, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#f1f5f9"},
    )

    return fig


# ─────────────────────────────────────────────
# Feature Range Hints (for input widgets)
# ─────────────────────────────────────────────

FEATURE_RANGES = {
    "Pregnancies":              (0,   17,   3),
    "Glucose":                  (44,  199,  117),
    "BloodPressure":            (24,  122,  72),
    "SkinThickness":            (7,   99,   29),
    "Insulin":                  (14,  846,  125),
    "BMI":                      (18,  67,   32),
    "DiabetesPedigreeFunction": (0.08, 2.42, 0.47),
    "Age":                      (21,  81,   33),
}
