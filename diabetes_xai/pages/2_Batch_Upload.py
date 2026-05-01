"""
pages/2_Batch_Upload.py — Batch CSV upload for multi-patient diabetes risk prediction.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = Path(__file__).parent.parent

st.set_page_config(page_title="Batch Upload | Diabetes XAI", page_icon="📂", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family:'Inter',sans-serif; }
    .stApp { background:linear-gradient(135deg,#0a0e1a 0%,#0f172a 50%,#0d1b2e 100%); color:#f1f5f9; }
    [data-testid="stSidebar"] { background:linear-gradient(180deg,#0f172a 0%,#1e293b 100%); border-right:1px solid #334155; }
    [data-testid="stSidebar"] * { color:#e2e8f0 !important; }
    h1,h2,h3 { color:#f1f5f9 !important; }
    .stButton>button { background:linear-gradient(135deg,#4f46e5,#7c3aed)!important; color:white!important;
        border:none!important; border-radius:8px!important; font-weight:600!important; }
    hr { border-color:#334155!important; }
    .stat-card { background:linear-gradient(135deg,#1e293b,#162032); border:1px solid #334155;
        border-radius:12px; padding:1.25rem; text-align:center; }
    .stat-val { font-size:2rem; font-weight:700; }
    .stat-label { font-size:0.8rem; color:#94a3b8; text-transform:uppercase; letter-spacing:0.05em; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📂 Batch Upload")
    st.markdown(
        "Upload a CSV containing the same columns as the training data (minus `Outcome`). "
        "The app will predict risk for every patient row."
    )
    st.markdown("---")
    st.markdown("**Required columns:**")
    st.code(
        "Pregnancies, Glucose, BloodPressure,\n"
        "SkinThickness, Insulin, BMI,\n"
        "DiabetesPedigreeFunction, Age",
        language="text",
    )
    st.markdown(
        "<div style='background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.4);"
        "border-radius:8px;padding:0.75rem;font-size:0.8rem;color:#fca5a5;margin-top:1rem;'>"
        "⚠️ Research Use Only. Not a medical diagnosis.</div>",
        unsafe_allow_html=True,
    )

# ── Load models ───────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(BASE_DIR))
from utils import RAW_FEATURE_COLS, ZERO_IMPUTE_COLS, ALL_FEATURES, get_risk_tier


@st.cache_resource
def _load():
    """Load model artefacts once per session."""
    from utils import load_models
    return load_models()


@st.cache_data
def _load_medians():
    """Load training medians for inference-time imputation."""
    path = BASE_DIR / "models" / "medians.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


try:
    ensemble, rf_model, scaler, feature_names = _load()
    medians = _load_medians()
    models_ok = True
except FileNotFoundError as e:
    st.error(f"❌ {e}")
    models_ok = False

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='background:linear-gradient(135deg,#818cf8,#a78bfa);-webkit-background-clip:text;"
    "-webkit-text-fill-color:transparent;background-clip:text;'>📂 Batch Patient Analysis</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "Upload a CSV file with multiple patients to get simultaneous risk predictions for the entire cohort."
)

if not models_ok:
    st.stop()

# ── File uploader ─────────────────────────────────────────────────────────────
st.markdown("---")
uploaded = st.file_uploader(
    "Upload Patient CSV",
    type=["csv"],
    help="CSV must contain the 8 raw clinical feature columns.",
)

# ── Sample template download ──────────────────────────────────────────────────
sample_data = pd.DataFrame(
    [
        [1, 85, 66, 29, 0, 26.6, 0.351, 31],
        [8, 183, 64, 0, 0, 23.3, 0.672, 32],
        [1, 89, 66, 23, 94, 28.1, 0.167, 21],
        [0, 137, 40, 35, 168, 43.1, 2.288, 33],
        [5, 116, 74, 0, 0, 25.6, 0.201, 30],
    ],
    columns=RAW_FEATURE_COLS,
)

csv_template = sample_data.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download CSV Template",
    data=csv_template,
    file_name="batch_template.csv",
    mime="text/csv",
)

# ── Process upload ────────────────────────────────────────────────────────────
if uploaded is not None:
    try:
        df_raw = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"❌ Could not read CSV: {exc}")
        st.stop()

    # ── Column validation ─────────────────────────────────────────────────────
    missing_cols = [c for c in RAW_FEATURE_COLS if c not in df_raw.columns]
    extra_cols = [c for c in df_raw.columns if c not in RAW_FEATURE_COLS + ["Outcome"]]

    if missing_cols:
        st.error(
            f"❌ The uploaded CSV is missing required columns: **{', '.join(missing_cols)}**\n\n"
            f"Expected columns: `{', '.join(RAW_FEATURE_COLS)}`"
        )
        st.stop()

    if extra_cols:
        st.warning(
            f"ℹ️ Unexpected columns found and will be ignored: {', '.join(extra_cols)}"
        )

    # Drop Outcome if present (not needed for inference)
    df_work = df_raw[RAW_FEATURE_COLS].copy()

    st.success(f"✅ Loaded **{len(df_work)} patients**. Running predictions…")

    # ── Preprocessing ─────────────────────────────────────────────────────────
    df_proc = df_work.copy()

    # Replace impossible zeros with NaN, impute with training medians
    for col in ZERO_IMPUTE_COLS:
        if col in df_proc.columns:
            df_proc[col] = df_proc[col].replace(0, np.nan)
    for col, med in medians.items():
        if col in df_proc.columns:
            df_proc[col] = df_proc[col].fillna(med)

    # Feature engineering
    df_proc["GlucoseBMI"] = df_proc["Glucose"] * df_proc["BMI"] / 100.0
    df_proc["AgeRisk"] = df_proc["Age"] * df_proc["DiabetesPedigreeFunction"]
    df_proc["InsulinResistance"] = df_proc["Insulin"] / (df_proc["BMI"] + 1.0)

    X_batch = scaler.transform(df_proc[ALL_FEATURES])

    # ── Predict ───────────────────────────────────────────────────────────────
    probs = ensemble.predict_proba(X_batch)[:, 1]
    risk_tiers = [get_risk_tier(p) for p in probs]
    risk_labels = [r[0] for r in risk_tiers]
    risk_colours = [r[1] for r in risk_tiers]

    # Build results dataframe
    results_df = df_work.copy()
    results_df["Risk_Probability"] = (probs * 100).round(1)
    results_df["Risk_Level"] = risk_labels

    # ── Summary Statistics ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 Cohort Summary")

    counts = results_df["Risk_Level"].value_counts().to_dict()
    low_n = counts.get("Low Risk", 0)
    border_n = counts.get("Borderline Risk", 0)
    high_n = counts.get("High Risk", 0)
    total = len(results_df)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"<div class='stat-card'><div class='stat-val' style='color:#22c55e'>{low_n}</div>"
            f"<div class='stat-label'>Low Risk</div></div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"<div class='stat-card'><div class='stat-val' style='color:#f97316'>{border_n}</div>"
            f"<div class='stat-label'>Borderline Risk</div></div>",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"<div class='stat-card'><div class='stat-val' style='color:#ef4444'>{high_n}</div>"
            f"<div class='stat-label'>High Risk</div></div>",
            unsafe_allow_html=True,
        )
    with c4:
        mean_risk = probs.mean()
        st.markdown(
            f"<div class='stat-card'><div class='stat-val' style='color:#818cf8'>{mean_risk:.1%}</div>"
            f"<div class='stat-label'>Mean Risk</div></div>",
            unsafe_allow_html=True,
        )

    # ── Bar chart summary ─────────────────────────────────────────────────────
    st.markdown("---")
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        bar_fig = go.Figure(go.Bar(
            x=["Low Risk", "Borderline Risk", "High Risk"],
            y=[low_n, border_n, high_n],
            marker_color=["#22c55e", "#f97316", "#ef4444"],
            text=[low_n, border_n, high_n],
            textposition="outside",
        ))
        bar_fig.update_layout(
            title="Risk Distribution",
            xaxis_title="Risk Tier",
            yaxis_title="Number of Patients",
            paper_bgcolor="#0f172a",
            plot_bgcolor="#1e293b",
            font=dict(color="#f1f5f9"),
            height=350,
        )
        st.plotly_chart(bar_fig, width="stretch")

    with chart_col2:
        pie_fig = go.Figure(go.Pie(
            labels=["Low Risk", "Borderline Risk", "High Risk"],
            values=[low_n, border_n, high_n],
            marker_colors=["#22c55e", "#f97316", "#ef4444"],
            hole=0.45,
            textinfo="percent+label",
        ))
        pie_fig.update_layout(
            title="Risk Proportion",
            paper_bgcolor="#0f172a",
            font=dict(color="#f1f5f9"),
            height=350,
        )
        st.plotly_chart(pie_fig, width="stretch")

    # ── Risk probability histogram ────────────────────────────────────────────
    hist_fig = px.histogram(
        x=probs * 100,
        nbins=20,
        color_discrete_sequence=["#818cf8"],
        labels={"x": "Risk Probability (%)"},
        title="Distribution of Risk Probabilities",
    )
    hist_fig.update_traces(marker_line_color="#4f46e5", marker_line_width=1)
    hist_fig.update_layout(
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        font=dict(color="#f1f5f9"),
        height=300,
    )
    st.plotly_chart(hist_fig, width="stretch")

    # ── Interactive results table ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📋 Patient Results Table")

    def _colour_risk(val: str) -> str:
        """Return a CSS background colour style for the risk tier cell."""
        mapping = {
            "Low Risk": "background-color: rgba(34,197,94,0.2); color: #22c55e; font-weight:600",
            "Borderline Risk": "background-color: rgba(249,115,22,0.2); color: #f97316; font-weight:600",
            "High Risk": "background-color: rgba(239,68,68,0.2); color: #ef4444; font-weight:600",
        }
        return mapping.get(val, "")

    styled = results_df.style.map(_colour_risk, subset=["Risk_Level"])
    st.dataframe(styled, width="stretch", height=400)

    # ── Export ────────────────────────────────────────────────────────────────
    st.markdown("---")
    csv_out = results_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Results CSV",
        data=csv_out,
        file_name="batch_predictions.csv",
        mime="text/csv",
        width="content",
    )
