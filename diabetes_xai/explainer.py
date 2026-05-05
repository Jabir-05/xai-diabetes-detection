"""
explainer.py — SHAP-based explainability and counterfactual logic.

Provides:
    get_shap_values()         — compute SHAP values for a single sample
    plot_waterfall()          — matplotlib waterfall figure
    plot_force()              — SHAP force plot HTML string
    get_shap_background()     — cached SHAP background summary array
    get_global_shap_values()  — SHAP values over training set (for beeswarm)
    plot_beeswarm()           — matplotlib beeswarm figure
    plot_bar()                — matplotlib bar plot of mean |SHAP|
    top_n_features()          — top-n features by mean |SHAP|
"""

from __future__ import annotations

import warnings
import os
import tempfile
from pathlib import Path

BASE_DIR = Path(__file__).parent
MPL_CACHE_DIR = Path(tempfile.gettempdir()) / "diabetes_xai_matplotlib"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for Streamlit
import matplotlib.pyplot as plt
import numpy as np
import shap
import joblib

MODELS_DIR = BASE_DIR / "models"


def _positive_class_values(shap_values):
    """Normalize SHAP outputs across old/new SHAP binary-class formats."""
    if isinstance(shap_values, list):
        return np.asarray(shap_values[1])

    values = np.asarray(shap_values)
    if values.ndim == 3 and values.shape[-1] == 2:
        return values[:, :, 1]
    return values


def _positive_expected_value(expected_value) -> float:
    """Return the positive-class expected value when SHAP exposes one value per class."""
    values = np.asarray(expected_value)
    if values.ndim > 0 and values.shape[0] >= 2:
        return float(values[1])
    return float(values)


def _shap_values(explainer, X: np.ndarray):
    """Compute SHAP values with compatibility for SHAP versions with additivity checks."""
    try:
        return explainer.shap_values(X, check_additivity=False)
    except TypeError:
        return explainer.shap_values(X)


# ─────────────────────────────────────────────
# SHAP explainer initialisation
# ─────────────────────────────────────────────

def get_explainer(rf_model, X_background: np.ndarray):
    """
    Build a shap.TreeExplainer for the standalone RF model.

    Parameters
    ----------
    rf_model : RandomForestClassifier
        Fitted random forest.
    X_background : np.ndarray
        Background training data for SHAP.

    Returns
    -------
    shap.TreeExplainer
    """
    explainer = shap.TreeExplainer(
        rf_model,
        data=shap.sample(X_background, min(100, len(X_background))),
        feature_perturbation="interventional",
    )
    return explainer


def get_shap_values(explainer, X_instance: np.ndarray) -> np.ndarray:
    """
    Compute SHAP values for a single instance.

    Parameters
    ----------
    explainer : shap.TreeExplainer
        Fitted SHAP explainer.
    X_instance : np.ndarray
        Shape (1, n_features) scaled input.

    Returns
    -------
    np.ndarray
        1-D array of SHAP values for the positive class.
    """
    shap_vals = _positive_class_values(_shap_values(explainer, X_instance))
    return shap_vals[0]


# ─────────────────────────────────────────────
# Waterfall plot
# ─────────────────────────────────────────────

def plot_waterfall(
    explainer,
    X_instance: np.ndarray,
    feature_names: list[str],
    max_display: int = 11,
) -> plt.Figure:
    """
    Create a SHAP waterfall plot for a single patient.

    Parameters
    ----------
    explainer : shap.TreeExplainer
    X_instance : np.ndarray  Shape (1, n_features)
    feature_names : list[str]
    max_display : int  Maximum number of features to display.

    Returns
    -------
    matplotlib.figure.Figure
    """
    shap_vals = _positive_class_values(_shap_values(explainer, X_instance))
    sv = shap_vals[0]
    base_val = _positive_expected_value(explainer.expected_value)

    expl_obj = shap.Explanation(
        values=sv,
        base_values=base_val,
        data=X_instance[0],
        feature_names=feature_names,
    )

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shap.plots.waterfall(expl_obj, max_display=max_display, show=False)

    fig = plt.gcf()
    fig.patch.set_facecolor("#0f172a")
    for ax_ in fig.axes:
        ax_.set_facecolor("#0f172a")
        ax_.tick_params(colors="#94a3b8")
        ax_.xaxis.label.set_color("#94a3b8")
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# Force plot (HTML)
# ─────────────────────────────────────────────

def plot_force_html(
    explainer,
    X_instance: np.ndarray,
    feature_names: list[str],
) -> str:
    """
    Generate an HTML SHAP force plot for a single patient.

    Parameters
    ----------
    explainer : shap.TreeExplainer
    X_instance : np.ndarray  Shape (1, n_features)
    feature_names : list[str]

    Returns
    -------
    str
        HTML string that can be embedded with st.components.v1.html().
    """
    shap_vals = _positive_class_values(_shap_values(explainer, X_instance))
    sv = shap_vals[0]
    base_val = _positive_expected_value(explainer.expected_value)

    shap.initjs()
    force_plot = shap.force_plot(
        base_val,
        sv,
        X_instance[0],
        feature_names=feature_names,
        matplotlib=False,
        show=False,
    )
    return f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"


# ─────────────────────────────────────────────
# Global SHAP (beeswarm + bar)
# ─────────────────────────────────────────────

def get_global_shap_values(
    explainer,
    X_background: np.ndarray,
    n_samples: int = 200,
) -> np.ndarray:
    """
    Compute SHAP values for a subset of training data.

    Parameters
    ----------
    explainer : shap.TreeExplainer
    X_background : np.ndarray  Full training set.
    n_samples : int  Number of samples to use.

    Returns
    -------
    np.ndarray  Shape (n_samples, n_features).
    """
    idx = np.random.RandomState(42).choice(
        len(X_background), size=min(n_samples, len(X_background)), replace=False
    )
    X_subset = X_background[idx]
    shap_vals = _positive_class_values(_shap_values(explainer, X_subset))
    return shap_vals, X_subset


def plot_beeswarm(
    shap_matrix: np.ndarray,
    X_subset: np.ndarray,
    feature_names: list[str],
    max_display: int = 11,
) -> plt.Figure:
    """
    Create a SHAP beeswarm plot over the training set.

    Parameters
    ----------
    shap_matrix : np.ndarray  Shape (n_samples, n_features)
    X_subset : np.ndarray  Corresponding feature values.
    feature_names : list[str]
    max_display : int

    Returns
    -------
    matplotlib.figure.Figure
    """
    expl = shap.Explanation(
        values=shap_matrix,
        data=X_subset,
        feature_names=feature_names,
    )
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0f172a")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shap.plots.beeswarm(expl, max_display=max_display, show=False)

    fig = plt.gcf()
    fig.patch.set_facecolor("#0f172a")
    for ax_ in fig.axes:
        ax_.set_facecolor("#0f172a")
        ax_.tick_params(colors="#94a3b8")
    plt.tight_layout()
    return fig


def plot_shap_bar(
    shap_matrix: np.ndarray,
    feature_names: list[str],
    max_display: int = 11,
) -> plt.Figure:
    """
    Bar chart of mean absolute SHAP values per feature (sorted descending).

    Parameters
    ----------
    shap_matrix : np.ndarray  Shape (n_samples, n_features)
    feature_names : list[str]
    max_display : int

    Returns
    -------
    matplotlib.figure.Figure
    """
    mean_abs = np.abs(shap_matrix).mean(axis=0)
    sorted_idx = np.argsort(mean_abs)[::-1][:max_display]
    sorted_names = [feature_names[i] for i in sorted_idx]
    sorted_vals = mean_abs[sorted_idx]

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")

    colours = plt.cm.plasma(np.linspace(0.3, 0.9, len(sorted_idx)))
    bars = ax.barh(sorted_names[::-1], sorted_vals[::-1], color=colours[::-1])
    ax.set_xlabel("Mean |SHAP value|", color="#94a3b8")
    ax.set_title("Global Feature Importance", color="#f1f5f9", fontsize=14)
    ax.tick_params(colors="#94a3b8")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#334155")
    ax.spines["bottom"].set_color("#334155")

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# Top-N feature helper
# ─────────────────────────────────────────────

def top_n_features(
    shap_matrix: np.ndarray,
    feature_names: list[str],
    n: int = 4,
) -> list[str]:
    """
    Return the top-n feature names ordered by mean |SHAP value|.

    Parameters
    ----------
    shap_matrix : np.ndarray
    feature_names : list[str]
    n : int

    Returns
    -------
    list[str]
    """
    mean_abs = np.abs(shap_matrix).mean(axis=0)
    sorted_idx = np.argsort(mean_abs)[::-1][:n]
    return [feature_names[i] for i in sorted_idx]
