"""
model.py — Training script for the Diabetes Risk XAI ensemble model.

Run standalone with:
    python model.py

Trains a soft-voting ensemble of four classifiers:
    Random Forest + Gradient Boosting + HistGradientBoosting + Logistic Regression

Saves:
    models/ensemble_model.pkl
    models/rf_model.pkl
    models/scaler.pkl
    models/feature_names.pkl
    models/metrics.json
    models/training_data.pkl   ← scaled X_train + y_train for SHAP background

Note: sklearn HistGradientBoostingClassifier is used instead of LightGBM
because LightGBM requires libomp (OpenMP runtime) which may not be
present on all macOS systems without Homebrew.
"""

from __future__ import annotations

import json
import os
import pickle
import urllib.request
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier,
    VotingClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

DATASET_PATH = DATA_DIR / "diabetes.csv"
DATASET_URL = (
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
    "pima-indians-diabetes.csv"
)

COLUMN_NAMES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome",
]

ZERO_IMPUTE_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
RAW_FEATURE_COLS = COLUMN_NAMES[:-1]
ENGINEERED_FEATURES = ["GlucoseBMI", "AgeRisk", "InsulinResistance"]
ALL_FEATURES = RAW_FEATURE_COLS + ENGINEERED_FEATURES


# ─────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────

def load_dataset() -> pd.DataFrame:
    """
    Load the PIMA Indians Diabetes dataset.

    Downloads from the internet if the local file is not found.

    Returns
    -------
    pd.DataFrame
        Raw dataset with column names assigned.
    """
    if not DATASET_PATH.exists():
        print(f"Downloading dataset from {DATASET_URL} …")
        urllib.request.urlretrieve(DATASET_URL, DATASET_PATH)
        print(f"Saved to {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH, header=None, names=COLUMN_NAMES)
    print(f"Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


# ─────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────

def preprocess_training(df: pd.DataFrame):
    """
    Preprocess training data: impute zeros, engineer features, scale.

    Parameters
    ----------
    df : pd.DataFrame
        Raw PIMA dataset with Outcome column.

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test, scaler, feature_names, medians)
    """
    df = df.copy()

    # Replace biologically impossible zeros with NaN
    for col in ZERO_IMPUTE_COLS:
        df[col] = df[col].replace(0, np.nan)

    # Median imputation (computed on full dataset before split is fine for demo;
    # in production compute only on train split)
    medians: dict = {}
    for col in ZERO_IMPUTE_COLS:
        med = df[col].median()
        medians[col] = med
        df[col] = df[col].fillna(med)

    # Feature engineering
    df["GlucoseBMI"] = df["Glucose"] * df["BMI"] / 100.0
    df["AgeRisk"] = df["Age"] * df["DiabetesPedigreeFunction"]
    df["InsulinResistance"] = df["Insulin"] / (df["BMI"] + 1.0)

    X = df[ALL_FEATURES]
    y = df["Outcome"]

    feature_names: list[str] = list(X.columns)

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scaling
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    return X_train_s, X_test_s, y_train.values, y_test.values, scaler, feature_names, medians


# ─────────────────────────────────────────────
# Model training
# ─────────────────────────────────────────────

def build_ensemble(X_train, y_train) -> VotingClassifier:
    """
    Build and train a soft-voting ensemble of four sklearn-native classifiers:

    1. RandomForestClassifier           (n_estimators=200)
    2. GradientBoostingClassifier       (n_estimators=200, scikit-learn native)
    3. HistGradientBoostingClassifier   (fast histogram-based gradient boosting)
    4. LogisticRegression               (max_iter=1000)

    All four are sklearn-native to guarantee API compatibility across versions.

    Parameters
    ----------
    X_train : np.ndarray
        Scaled training features.
    y_train : np.ndarray
        Training labels.

    Returns
    -------
    VotingClassifier
        Fitted soft-voting ensemble.
    """
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    gb = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=4,
        random_state=42,
    )
    hgb = HistGradientBoostingClassifier(
        max_iter=200,
        random_state=42,
    )
    lr = LogisticRegression(max_iter=1000, random_state=42)

    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("gb", gb), ("hgb", hgb), ("lr", lr)],
        voting="soft",
        n_jobs=-1,
    )
    print("Training ensemble model …")
    ensemble.fit(X_train, y_train)
    return ensemble


def build_rf(X_train, y_train) -> RandomForestClassifier:
    """
    Train a standalone Random Forest for SHAP explainability.

    Parameters
    ----------
    X_train : np.ndarray
        Scaled training features.
    y_train : np.ndarray
        Training labels.

    Returns
    -------
    RandomForestClassifier
        Fitted random forest model.
    """
    print("Training standalone RF model (for SHAP) …")
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    return rf


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────

def evaluate(model, X_test, y_test) -> dict:
    """
    Evaluate a classifier and print results to console.

    Parameters
    ----------
    model : sklearn estimator
        Fitted model with predict / predict_proba methods.
    X_test : np.ndarray
        Scaled test features.
    y_test : np.ndarray
        True test labels.

    Returns
    -------
    dict
        Dictionary of metric name → float value.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)

    print("\n" + "=" * 55)
    print("ENSEMBLE MODEL — TEST SET PERFORMANCE")
    print("=" * 55)
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return {"accuracy": acc, "roc_auc": auc, "f1": f1, "precision": prec, "recall": rec}


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    """
    Full training pipeline: load → preprocess → train → evaluate → save.
    """
    # 1. Load data
    df = load_dataset()

    # 2. Preprocess
    X_train, X_test, y_train, y_test, scaler, feature_names, medians = (
        preprocess_training(df)
    )

    # 3. Train models
    ensemble = build_ensemble(X_train, y_train)
    rf_model = build_rf(X_train, y_train)

    # 4. Evaluate
    metrics = evaluate(ensemble, X_test, y_test)
    metrics["dataset_size"] = len(df)

    # 5. Save artefacts
    joblib.dump(ensemble, MODELS_DIR / "ensemble_model.pkl")
    print("Saved → models/ensemble_model.pkl")

    joblib.dump(rf_model, MODELS_DIR / "rf_model.pkl")
    print("Saved → models/rf_model.pkl")

    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    print("Saved → models/scaler.pkl")

    with open(MODELS_DIR / "feature_names.pkl", "wb") as f:
        pickle.dump(feature_names, f)
    print("Saved → models/feature_names.pkl")

    # Save training data for SHAP background (store original X_train array)
    joblib.dump((X_train, y_train), MODELS_DIR / "training_data.pkl")
    print("Saved → models/training_data.pkl")

    # Save medians for inference-time imputation
    with open(MODELS_DIR / "medians.json", "w") as f:
        json.dump(medians, f, indent=2)
    print("Saved → models/medians.json")

    # Save metrics for the dashboard
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved → models/metrics.json")

    print("\n✅ All artefacts saved to models/")
    print("Run the app with:  streamlit run app.py")


if __name__ == "__main__":
    main()
