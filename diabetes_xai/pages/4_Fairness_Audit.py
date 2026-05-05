"""
pages/4_Fairness_Audit.py — Fairlearn bias audit using age group as sensitive feature.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).parent.parent

st.set_page_config(page_title="Fairness Audit | Diabetes XAI", page_icon="⚖️", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family:'Inter',sans-serif; }
    .stApp { background:linear-gradient(135deg,#0a0e1a 0%,#0f172a 50%,#0d1b2e 100%); color:#f1f5f9; }
    [data-testid="stSidebar"] { background:linear-gradient(180deg,#0f172a 0%,#1e293b 100%); border-right:1px solid #334155; }
    [data-testid="stSidebar"] * { color:#e2e8f0 !important; }
    h1,h2,h3 { color:#f1f5f9 !important; }
    hr { border-color:#334155!important; }
    .metric-card { background:linear-gradient(135deg,#1e293b,#162032); border:1px solid #334155;
        border-radius:12px; padding:1.25rem; text-align:center; }
    .metric-val { font-size:1.8rem; font-weight:700; }
    .metric-label { font-size:0.8rem; color:#94a3b8; text-transform:uppercase; letter-spacing:0.05em; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚖️ Fairness Audit")
    st.markdown(
        "This page audits the model for demographic bias by comparing performance "
        "metrics across age groups using **Fairlearn**."
    )
    st.markdown("---")
    st.markdown("**Sensitive attribute:** Age group")
    st.markdown("- `<30` years\n- `30–50` years\n- `>50` years")
    st.markdown("---")
    st.markdown(
        "<div style='background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.4);"
        "border-radius:8px;padding:0.75rem;font-size:0.8rem;color:#fca5a5;'>"
        "⚠️ Research Use Only. Not a medical diagnosis.</div>",
        unsafe_allow_html=True,
    )

# ── Load resources ────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(BASE_DIR))
from utils import ALL_FEATURES, ZERO_IMPUTE_COLS, DATA_DIR


@st.cache_resource
def _load():
    """Load model artefacts once per session."""
    from utils import load_models
    return load_models()


@st.cache_data
def _prepare_test_data():
    """
    Rebuild the test split identically to model.py so we have ages and labels.

    Returns
    -------
    tuple: (X_test_scaled, y_test, ages_test) or (None, None, None) on failure.
    """
    import json
    import numpy as np
    from sklearn.model_selection import train_test_split

    csv_path = DATA_DIR / "diabetes.csv"
    medians_path = BASE_DIR / "models" / "medians.json"

    if not csv_path.exists() or not medians_path.exists():
        return None, None, None

    try:
        from model import COLUMN_NAMES, ZERO_IMPUTE_COLS as ZIC
        df = pd.read_csv(csv_path, header=None, names=COLUMN_NAMES)

        with open(medians_path) as f:
            medians = json.load(f)

        # Replicate preprocessing from model.py
        for col in ZIC:
            df[col] = df[col].replace(0, np.nan)
        for col, med in medians.items():
            df[col] = df[col].fillna(med)

        df["GlucoseBMI"] = df["Glucose"] * df["BMI"] / 100.0
        df["AgeRisk"] = df["Age"] * df["DiabetesPedigreeFunction"]
        df["InsulinResistance"] = df["Insulin"] / (df["BMI"] + 1.0)

        X = df[ALL_FEATURES]
        y = df["Outcome"].values
        ages = df["Age"].values

        _, X_test, _, y_test, _, ages_test = train_test_split(
            X, y, ages, test_size=0.2, random_state=42, stratify=y
        )

        X_test_s = scaler.transform(X_test)
        return X_test_s, y_test, ages_test

    except Exception as exc:
        st.error(f"❌ Failed to prepare test data: {exc}")
        return None, None, None


try:
    ensemble, rf_model, scaler, feature_names = _load()
    models_ok = True
except FileNotFoundError as e:
    st.error(f"❌ {e}")
    models_ok = False

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='background:linear-gradient(135deg,#818cf8,#a78bfa);-webkit-background-clip:text;"
    "-webkit-text-fill-color:transparent;background-clip:text;'>⚖️ Fairness & Bias Audit</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "Auditing the ensemble model for demographic bias across age groups. "
    "A fair model should perform similarly regardless of a patient's age."
)

if not models_ok:
    st.stop()

# ── Prepare test data ─────────────────────────────────────────────────────────
with st.spinner("Preparing test data and running fairness audit…"):
    X_test, y_test, ages_test = _prepare_test_data()

if X_test is None:
    st.error(
        "❌ Could not load test data. Make sure `data/diabetes.csv` and "
        "`models/medians.json` exist (run `python model.py`)."
    )
    st.stop()

# ── Run fairness audit ────────────────────────────────────────────────────────
try:
    from fairness import run_fairness_audit, plot_fairness_bar, fairness_plain_english

    results = run_fairness_audit(ensemble, X_test, y_test, ages_test)
    metric_frame = results["metric_frame"]
    dp_diff = results["dp_diff"]
    eo_diff = results["eo_diff"]

except ImportError as exc:
    st.error(
        f"❌ Fairlearn import error: {exc}\n\n"
        "Install with: `pip install fairlearn>=0.10.0`"
    )
    st.stop()
except Exception as exc:
    st.error(f"❌ Fairness audit failed: {exc}")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Summary Metrics
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📊 Summary Metrics")

overall = metric_frame.overall
m1, m2, m3, m4 = st.columns(4)

def _colour_diff(val: float) -> str:
    return "#22c55e" if abs(val) < 0.05 else "#f97316" if abs(val) < 0.10 else "#ef4444"

with m1:
    st.markdown(
        f"<div class='metric-card'><div class='metric-val' style='color:#818cf8'>"
        f"{overall['accuracy']:.3f}</div>"
        f"<div class='metric-label'>Overall Accuracy</div></div>",
        unsafe_allow_html=True,
    )
with m2:
    st.markdown(
        f"<div class='metric-card'><div class='metric-val' style='color:#34d399'>"
        f"{overall['precision']:.3f}</div>"
        f"<div class='metric-label'>Overall Precision</div></div>",
        unsafe_allow_html=True,
    )
with m3:
    st.markdown(
        f"<div class='metric-card'><div class='metric-val' style='color:{_colour_diff(dp_diff)}'>"
        f"{dp_diff:+.4f}</div>"
        f"<div class='metric-label'>Demographic Parity Diff</div></div>",
        unsafe_allow_html=True,
    )
with m4:
    st.markdown(
        f"<div class='metric-card'><div class='metric-val' style='color:{_colour_diff(eo_diff)}'>"
        f"{eo_diff:+.4f}</div>"
        f"<div class='metric-label'>Equalized Odds Diff</div></div>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Per-Group Metrics Table
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📋 Per-Age-Group Performance")

by_group = metric_frame.by_group.copy()
by_group.columns = [c.capitalize() for c in by_group.columns]
by_group.index.name = "Age Group"
by_group = by_group.round(4)

# Highlight min accuracy row
def _highlight_min(s):
    is_min = s == s.min()
    return ["background-color: rgba(239,68,68,0.2); color:#ef4444" if v else "" for v in is_min]

styled_table = by_group.style.apply(_highlight_min, subset=["Accuracy"])
st.dataframe(styled_table, width="stretch")

# ─────────────────────────────────────────────────────────────────────────────
# Bar Chart — Per-Group Metrics
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📊 Per-Group Metrics Comparison")

bar_fig = plot_fairness_bar(metric_frame)
st.plotly_chart(bar_fig, width="stretch")

# ─────────────────────────────────────────────────────────────────────────────
# Age Group Distribution
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 👥 Test Set Age Group Composition")

from fairness import bin_age, AGE_LABELS
import plotly.graph_objects as go

age_groups = bin_age(ages_test)
group_counts = pd.Series(age_groups).value_counts().reindex(AGE_LABELS, fill_value=0)

dist_fig = go.Figure(go.Bar(
    x=group_counts.index.tolist(),
    y=group_counts.values,
    marker_color=["#818cf8", "#34d399", "#f97316"],
    text=group_counts.values,
    textposition="outside",
))
dist_fig.update_layout(
    title="Test Set Distribution by Age Group",
    xaxis_title="Age Group",
    yaxis_title="Number of Patients",
    paper_bgcolor="#0f172a",
    plot_bgcolor="#1e293b",
    font=dict(color="#f1f5f9"),
    height=320,
)
st.plotly_chart(dist_fig, width="stretch")

# ─────────────────────────────────────────────────────────────────────────────
# Plain-English Interpretation
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🗣️ Plain-English Interpretation")

interpretation = fairness_plain_english(metric_frame, dp_diff, eo_diff)
st.markdown(interpretation)

# ─────────────────────────────────────────────────────────────────────────────
# Fairness Thresholds Reference
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("ℹ️ Fairness Metrics Reference"):
    st.markdown(
        """
| Metric | Definition | Ideal Value | Threshold |
|--------|-----------|-------------|-----------|
| **Demographic Parity Difference** | Difference in positive prediction rates between groups | 0.00 | < 0.05 |
| **Equalized Odds Difference** | Max difference in TPR or FPR across groups | 0.00 | < 0.05 |
| **Accuracy Parity** | Equal accuracy across all groups | Equal | Δ < 5% |

> Values closer to **0** indicate less bias. A threshold of **0.05** is commonly used in industry fairness standards.
        """
    )
