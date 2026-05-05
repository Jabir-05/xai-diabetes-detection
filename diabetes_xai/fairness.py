"""
fairness.py — Fairlearn bias audit for the Diabetes Risk model.

Computes per-group metrics (accuracy, precision, recall) using Age group
as the sensitive attribute and displays results in Streamlit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from fairlearn.metrics import MetricFrame, demographic_parity_difference, equalized_odds_difference
from sklearn.metrics import accuracy_score, precision_score, recall_score


# Age group binning
AGE_BINS = [0, 30, 51, 200]
AGE_LABELS = ["<30", "30–50", ">50"]


def bin_age(ages: np.ndarray) -> np.ndarray:
    """
    Bin an array of ages into three demographic groups.

    Parameters
    ----------
    ages : np.ndarray
        Array of age values.

    Returns
    -------
    np.ndarray
        String array with group labels "<30", "30–50", ">50".
    """
    age_series = pd.Series(np.asarray(ages, dtype=float), name="Age")
    return (
        pd.cut(age_series, bins=AGE_BINS, labels=AGE_LABELS, right=False)
        .astype(str)
        .to_numpy()
    )


def run_fairness_audit(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    test_ages: np.ndarray,
) -> dict:
    """
    Run a fairness audit using Fairlearn MetricFrame.

    Parameters
    ----------
    model : sklearn estimator
        Fitted model (ensemble or RF).
    X_test : np.ndarray
        Scaled test features.
    y_test : np.ndarray
        True test labels.
    test_ages : np.ndarray
        Age values for the test set (unscaled original ages).

    Returns
    -------
    dict with keys:
        "metric_frame"  : fairlearn.metrics.MetricFrame
        "dp_diff"       : float — demographic parity difference
        "eo_diff"       : float — equalized odds difference
        "age_groups"    : np.ndarray — group labels per test sample
        "y_pred"        : np.ndarray — model predictions
    """
    y_pred = model.predict(X_test)
    age_groups = bin_age(test_ages)

    metric_frame = MetricFrame(
        metrics={
            "accuracy": accuracy_score,
            "precision": lambda y_true, y_pred: precision_score(y_true, y_pred, zero_division=0),
            "recall": lambda y_true, y_pred: recall_score(y_true, y_pred, zero_division=0),
        },
        y_true=y_test,
        y_pred=y_pred,
        sensitive_features=age_groups,
    )

    dp_diff = demographic_parity_difference(
        y_test, y_pred, sensitive_features=age_groups
    )
    eo_diff = equalized_odds_difference(
        y_test, y_pred, sensitive_features=age_groups
    )

    return {
        "metric_frame": metric_frame,
        "dp_diff": dp_diff,
        "eo_diff": eo_diff,
        "age_groups": age_groups,
        "y_pred": y_pred,
    }


def plot_fairness_bar(metric_frame: MetricFrame) -> go.Figure:
    """
    Create a grouped bar chart showing per-group metrics side by side.

    Parameters
    ----------
    metric_frame : fairlearn.metrics.MetricFrame

    Returns
    -------
    plotly.graph_objects.Figure
    """
    by_group = metric_frame.by_group
    groups = by_group.index.tolist()
    metrics = by_group.columns.tolist()

    colours = {
        "accuracy": "#818cf8",
        "precision": "#34d399",
        "recall": "#f97316",
    }

    fig = go.Figure()
    for metric in metrics:
        fig.add_trace(go.Bar(
            name=metric.capitalize(),
            x=groups,
            y=by_group[metric].values,
            marker_color=colours.get(metric, "#94a3b8"),
            text=[f"{v:.3f}" for v in by_group[metric].values],
            textposition="outside",
        ))

    # Overall reference lines
    overall = metric_frame.overall
    for metric in metrics:
        fig.add_hline(
            y=overall[metric],
            line_dash="dot",
            line_color=colours.get(metric, "#94a3b8"),
            opacity=0.5,
            annotation_text=f"Overall {metric}: {overall[metric]:.3f}",
            annotation_position="top right",
        )

    fig.update_layout(
        title="Per-Age-Group Model Performance",
        barmode="group",
        xaxis_title="Age Group",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1.1]),
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        font=dict(color="#f1f5f9"),
        legend=dict(bgcolor="#1e293b"),
        height=420,
    )
    return fig


def fairness_plain_english(metric_frame: MetricFrame, dp_diff: float, eo_diff: float) -> str:
    """
    Generate a plain-English interpretation of the fairness audit.

    Parameters
    ----------
    metric_frame : fairlearn.metrics.MetricFrame
    dp_diff : float
    eo_diff : float

    Returns
    -------
    str
        Human-readable summary.
    """
    by_group = metric_frame.by_group
    overall_acc = metric_frame.overall["accuracy"]

    worst_group = by_group["accuracy"].idxmin()
    worst_acc = by_group["accuracy"].min()
    diff_pct = (overall_acc - worst_acc) * 100

    lines = [
        f"**Fairness Audit Summary**",
        f"",
        f"- The model achieves an overall accuracy of **{overall_acc:.1%}**.",
        f"- The **{worst_group}** age group shows the lowest accuracy "
        f"(**{worst_acc:.1%}**), which is **{diff_pct:.1f}% below** the overall average.",
        f"- **Demographic Parity Difference**: {dp_diff:.4f}  "
        f"(0 = perfectly fair; values closer to 0 are better).",
        f"- **Equalized Odds Difference**: {eo_diff:.4f}.",
    ]

    if abs(dp_diff) < 0.05 and abs(eo_diff) < 0.05:
        lines.append(
            "✅ Both fairness metrics are within acceptable bounds (< 0.05), "
            "indicating low demographic bias."
        )
    else:
        lines.append(
            "⚠️ One or more fairness metrics exceed the 0.05 threshold. "
            "Consider bias-mitigation techniques (re-weighting, adversarial de-biasing) "
            "before clinical deployment."
        )

    return "\n".join(lines)
