"""
pages/1_Single_Patient.py — Single patient prediction with full SHAP + LLM + PDF flow.
"""

from __future__ import annotations

import json
import traceback
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

BASE_DIR = Path(__file__).parent.parent

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(page_title="Single Patient | Diabetes XAI", page_icon="🔬", layout="wide")

# ─────────────────────────────────────────────
# Inline CSS (inherits from app.py via shared session, but we re-declare essentials)
# ─────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --ink:#122033;
        --muted:#64748b;
        --line:#dbe6f1;
        --blue:#2563eb;
        --teal:#0f9f8f;
        --green:#16a34a;
        --amber:#d97706;
        --red:#dc2626;
        --surface:#ffffff;
        --soft:#f5f9fc;
    }

    html, body, [class*="css"] {
        font-family:'Inter',system-ui,-apple-system,BlinkMacSystemFont,sans-serif;
        color:var(--ink);
    }

    .stApp {
        background:
            radial-gradient(circle at 12% 0%, rgba(15,159,143,0.13), transparent 30rem),
            radial-gradient(circle at 90% 15%, rgba(37,99,235,0.10), transparent 28rem),
            linear-gradient(180deg,#f8fbff 0%,#eef5f9 100%);
        color:var(--ink);
    }

    [data-testid="stSidebar"] {
        background:linear-gradient(180deg,#10243c 0%,#173a52 100%);
        border-right:1px solid rgba(255,255,255,0.14);
    }

    [data-testid="stSidebar"] * { color:#eef7ff !important; }

    h1,h2,h3 {
        color:var(--ink) !important;
        letter-spacing:0;
    }

    p, li, label, span { letter-spacing:0; }
    hr { border-color:var(--line)!important; }

    .page-hero {
        background:linear-gradient(135deg,#ffffff 0%,#edf7f6 100%);
        border:1px solid var(--line);
        border-radius:8px;
        padding:1.45rem 1.55rem;
        margin-bottom:1rem;
        box-shadow:0 14px 34px rgba(18,32,51,0.08);
    }

    .hero-kicker {
        color:var(--teal);
        font-size:.76rem;
        font-weight:800;
        text-transform:uppercase;
        letter-spacing:.08em;
        margin-bottom:.25rem;
    }

    .hero-title {
        color:var(--ink);
        font-size:2.15rem;
        font-weight:800;
        margin:0;
    }

    .hero-copy {
        color:var(--muted);
        max-width:780px;
        margin:.4rem 0 0 0;
        font-size:.98rem;
        line-height:1.55;
    }

    .section-chip {
        display:inline-flex;
        align-items:center;
        gap:.4rem;
        background:#e9f4ff;
        border:1px solid #bfdbfe;
        color:#1d4ed8;
        border-radius:999px;
        padding:.24rem .72rem;
        font-size:.78rem;
        font-weight:800;
        margin-bottom:.35rem;
    }

    .risk-banner {
        background:#ffffff;
        border:1px solid var(--line);
        border-radius:8px;
        padding:1.2rem;
        margin:.6rem 0 1rem 0;
        box-shadow:0 10px 28px rgba(18,32,51,0.08);
    }

    .risk-label {
        color:var(--muted);
        font-size:.74rem;
        font-weight:800;
        text-transform:uppercase;
        letter-spacing:.07em;
    }

    .risk-value {
        font-size:2.15rem;
        font-weight:800;
        line-height:1.05;
        margin:.2rem 0;
    }

    .risk-subtext {
        color:var(--muted);
        font-size:.86rem;
        margin:0;
    }

    .snapshot-card,
    .driver-card,
    .action-card {
        background:#ffffff;
        border:1px solid var(--line);
        border-radius:8px;
        padding:1rem;
        min-height:108px;
        box-shadow:0 8px 22px rgba(18,32,51,0.06);
    }

    .snapshot-title,
    .driver-title {
        color:var(--muted);
        font-size:.72rem;
        font-weight:800;
        text-transform:uppercase;
        letter-spacing:.06em;
        margin-bottom:.4rem;
    }

    .snapshot-value {
        color:var(--ink);
        font-size:1.45rem;
        font-weight:800;
        line-height:1.05;
    }

    .status-pill {
        display:inline-flex;
        margin-top:.55rem;
        padding:.18rem .55rem;
        border-radius:999px;
        font-size:.74rem;
        font-weight:800;
    }

    .driver-value {
        font-size:1.08rem;
        font-weight:800;
        color:var(--ink);
    }

    .driver-note {
        color:var(--muted);
        font-size:.82rem;
        margin-top:.25rem;
    }

    .summary-box {
        background:#ffffff;
        border:1px solid var(--line);
        border-left:5px solid var(--teal);
        border-radius:8px;
        padding:1rem 1.1rem;
        color:var(--ink);
        box-shadow:0 8px 22px rgba(18,32,51,0.05);
    }

    .stButton>button,
    .stDownloadButton>button,
    [data-testid="stFormSubmitButton"] button {
        background:linear-gradient(135deg,var(--blue),var(--teal))!important;
        color:white!important;
        border:none!important;
        border-radius:8px!important;
        font-weight:800!important;
        min-height:2.8rem;
        box-shadow:0 12px 26px rgba(37,99,235,.20);
    }

    .stButton>button:hover,
    .stDownloadButton>button:hover,
    [data-testid="stFormSubmitButton"] button:hover {
        transform:translateY(-1px)!important;
        box-shadow:0 16px 30px rgba(15,159,143,.24)!important;
    }

    div[data-baseweb="input"] input,
    div[data-baseweb="select"] {
        color:var(--ink)!important;
    }

    .stAlert {
        color:var(--ink);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Single Patient")
    st.markdown(
        "Enter the patient's clinical measurements and click **Predict** to get a "
        "real-time diabetes risk assessment with SHAP explanations."
    )
    st.markdown("---")
    st.markdown(
        "<div style='background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.4);"
        "border-radius:8px;padding:0.75rem;font-size:0.8rem;color:#fca5a5;'>"
        "<b>Research Use Only.</b><br>Not a medical diagnosis.</div>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
# Load models (cached)
# ─────────────────────────────────────────────
@st.cache_resource
def _load():
    """Load all model artefacts once per session."""
    import sys
    sys.path.insert(0, str(BASE_DIR))
    from utils import load_models
    return load_models()


@st.cache_resource
def _load_training_data():
    """Load training data for SHAP background (cached)."""
    import joblib
    path = BASE_DIR / "models" / "training_data.pkl"
    if path.exists():
        return joblib.load(path)
    return None, None


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
    X_train_bg, _ = _load_training_data()
    medians = _load_medians()
    models_ok = True
except FileNotFoundError as e:
    st.error(f"❌ {e}")
    models_ok = False

# ─────────────────────────────────────────────
# SHAP Explainer (cached)
# ─────────────────────────────────────────────
@st.cache_resource
def _get_explainer():
    """Build SHAP TreeExplainer once per session."""
    import sys
    sys.path.insert(0, str(BASE_DIR))
    from explainer import get_explainer
    return get_explainer(rf_model, X_train_bg)


# ─────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state["history"] = []

# ─────────────────────────────────────────────
# Page Header
# ─────────────────────────────────────────────
st.markdown(
    """
    <div class="page-hero">
        <div class="hero-kicker">Clinical Decision Support Workspace</div>
        <h1 class="hero-title">Single Patient Assessment</h1>
        <p class="hero-copy">
            Enter patient measurements, generate a diabetes risk score, review key drivers,
            and export an explainable report for academic demonstration.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not models_ok:
    st.stop()

# ─────────────────────────────────────────────
# Input Form
# ─────────────────────────────────────────────
from utils import FEATURE_RANGES, get_risk_tier  # noqa: E402


def _hex_to_rgba(hex_colour: str, alpha: float = 0.12) -> str:
    """Convert a hex color to an rgba string."""
    hex_colour = hex_colour.lstrip("#")
    r, g, b = (int(hex_colour[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _clinical_status(feature: str, value: float) -> tuple[str, str]:
    """Return a simple display status for core clinical values."""
    if feature == "Glucose":
        if value < 100:
            return "In target", "#16a34a"
        if value < 126:
            return "Watch", "#d97706"
        return "High", "#dc2626"
    if feature == "BMI":
        if value < 25:
            return "Healthy", "#16a34a"
        if value < 30:
            return "Overweight", "#d97706"
        return "Obesity range", "#dc2626"
    if feature == "BloodPressure":
        if value < 80:
            return "In target", "#16a34a"
        if value < 90:
            return "Elevated", "#d97706"
        return "High", "#dc2626"
    if feature == "Age":
        if value < 40:
            return "Lower age risk", "#16a34a"
        if value < 55:
            return "Moderate age risk", "#d97706"
        return "Higher age risk", "#dc2626"
    return "Recorded", "#2563eb"


def _snapshot_card(title: str, value: str, status: str, colour: str) -> str:
    return f"""
    <div class="snapshot-card">
        <div class="snapshot-title">{title}</div>
        <div class="snapshot-value">{value}</div>
        <span class="status-pill" style="color:{colour}; background:{_hex_to_rgba(colour, .13)};
             border:1px solid {_hex_to_rgba(colour, .32)};">{status}</span>
    </div>
    """


def _feature_display_value(feature: str, values: dict) -> str:
    if feature == "GlucoseBMI":
        return f"{values['Glucose'] * values['BMI'] / 100.0:.1f}"
    if feature == "AgeRisk":
        return f"{values['Age'] * values['DiabetesPedigreeFunction']:.1f}"
    if feature == "InsulinResistance":
        return f"{values['Insulin'] / (values['BMI'] + 1.0):.1f}"
    raw = values.get(feature, "N/A")
    if isinstance(raw, float):
        return f"{raw:.2f}" if feature == "DiabetesPedigreeFunction" else f"{raw:.1f}"
    return str(raw)


def _driver_card(feature: str, contribution: float, values: dict) -> str:
    raises_risk = contribution > 0
    colour = "#dc2626" if raises_risk else "#16a34a"
    direction = "Raises model risk" if raises_risk else "Lowers model risk"
    sign = "+" if contribution > 0 else ""
    return f"""
    <div class="driver-card">
        <div class="driver-title">{direction}</div>
        <div class="driver-value">{feature}</div>
        <div class="driver-note">Value: <b>{_feature_display_value(feature, values)}</b></div>
        <span class="status-pill" style="color:{colour}; background:{_hex_to_rgba(colour, .13)};
             border:1px solid {_hex_to_rgba(colour, .32)};">SHAP {sign}{contribution:.4f}</span>
    </div>
    """


def _profile_radar(values: dict) -> go.Figure:
    radar_features = ["Glucose", "BMI", "BloodPressure", "Insulin", "Age"]
    scaled = []
    for feature in radar_features:
        low, high, _ = FEATURE_RANGES[feature]
        score = (float(values[feature]) - float(low)) / (float(high) - float(low))
        scaled.append(max(0, min(100, score * 100)))

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=scaled + [scaled[0]],
            theta=radar_features + [radar_features[0]],
            fill="toself",
            fillcolor="rgba(15,159,143,0.20)",
            line=dict(color="#0f9f8f", width=3),
            name="Patient profile",
        )
    )
    fig.update_layout(
        height=330,
        margin=dict(l=30, r=30, t=35, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#122033"),
        polar=dict(
            bgcolor="#ffffff",
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=10, color="#64748b")),
            angularaxis=dict(tickfont=dict(size=12, color="#122033")),
        ),
        showlegend=False,
    )
    return fig


def _professional_gauge(probability: float) -> go.Figure:
    risk_label, risk_colour = get_risk_tier(probability)
    pct = probability * 100
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pct,
            number={"suffix": "%", "font": {"size": 38, "color": "#122033"}},
            title={"text": f"<b>{risk_label}</b>", "font": {"size": 18, "color": risk_colour}},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": "#94a3b8",
                    "tickfont": {"color": "#475569", "size": 11},
                },
                "bar": {"color": risk_colour, "thickness": 0.30},
                "bgcolor": "#ffffff",
                "borderwidth": 1,
                "bordercolor": "#dbe6f1",
                "steps": [
                    {"range": [0, 35], "color": "rgba(22,163,74,0.16)"},
                    {"range": [35, 65], "color": "rgba(217,119,6,0.18)"},
                    {"range": [65, 100], "color": "rgba(220,38,38,0.16)"},
                ],
                "threshold": {
                    "line": {"color": "#122033", "width": 3},
                    "thickness": 0.8,
                    "value": pct,
                },
            },
        )
    )
    fig.update_layout(
        height=300,
        margin=dict(t=50, b=0, l=25, r=25),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _next_steps(values: dict, risk_label: str) -> list[str]:
    steps = []
    if values["Glucose"] >= 126:
        steps.append("Discuss fasting glucose or HbA1c confirmation with a clinician.")
    elif values["Glucose"] >= 100:
        steps.append("Monitor glucose trends and review diet timing before follow-up.")
    if values["BMI"] >= 30:
        steps.append("Consider a supervised weight-management and activity plan.")
    if values["BloodPressure"] >= 90:
        steps.append("Recheck blood pressure and evaluate cardiovascular risk factors.")
    if risk_label == "High Risk":
        steps.append("Prioritize a clinical consultation for complete diagnostic testing.")
    if not steps:
        steps.append("Maintain routine screening and healthy lifestyle habits.")
    return steps[:4]

st.markdown('<span class="section-chip">Patient clinical data</span>', unsafe_allow_html=True)
st.markdown("Use the fields below to run a single-patient risk assessment.")

with st.form("patient_form"):
    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input(
            "Pregnancies", min_value=0, max_value=20, value=3, step=1,
            help="Number of times pregnant"
        )
        glucose = st.number_input(
            "Glucose (mg/dL)", min_value=44, max_value=300, value=117, step=1,
            help="Plasma glucose concentration (2-hour oral glucose tolerance test)"
        )
        blood_pressure = st.number_input(
            "Blood Pressure (mm Hg)", min_value=0, max_value=150, value=72, step=1,
            help="Diastolic blood pressure"
        )
        skin_thickness = st.number_input(
            "Skin Thickness (mm)", min_value=0, max_value=120, value=29, step=1,
            help="Triceps skin fold thickness"
        )

    with col2:
        insulin = st.number_input(
            "Insulin (μU/mL)", min_value=0, max_value=900, value=125, step=1,
            help="2-Hour serum insulin"
        )
        bmi = st.number_input(
            "BMI (kg/m²)", min_value=10.0, max_value=80.0, value=32.0, step=0.1,
            help="Body mass index"
        )
        dpf = st.number_input(
            "Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.47, step=0.01,
            help="Genetic predisposition score based on family history"
        )
        age = st.number_input(
            "Age (years)", min_value=1, max_value=120, value=33, step=1,
            help="Patient age in years"
        )

    predict_btn = st.form_submit_button("Predict Diabetes Risk", width="stretch")

# ─────────────────────────────────────────────
# Prediction Flow
# ─────────────────────────────────────────────
if predict_btn:
    raw_input = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age,
    }

    # Preprocess
    import sys
    sys.path.insert(0, str(BASE_DIR))
    from utils import preprocess_with_median

    df_input = pd.DataFrame([raw_input])
    X_scaled = preprocess_with_median(df_input, medians, scaler)

    # Predict
    prob = ensemble.predict_proba(X_scaled)[0, 1]
    risk_label, risk_colour = get_risk_tier(prob)

    # ── Results layout ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<span class="section-chip">Assessment results</span>', unsafe_allow_html=True)

    res_col1, res_col2 = st.columns([1, 1.4])

    with res_col1:
        st.markdown(
            f"<div class='risk-banner' style='border-left:6px solid {risk_colour};'>"
            f"<div class='risk-label'>Current model prediction</div>"
            f"<div class='risk-value' style='color:{risk_colour}'>{risk_label}</div>"
            f"<p class='risk-subtext'><b style='font-size:1.25rem;color:#122033'>{prob:.1%}</b> estimated probability of diabetes.</p>"
            f"<p class='risk-subtext'>This result is generated by the ensemble model and should be interpreted with SHAP explanations below.</p>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with res_col2:
        gauge_fig = _professional_gauge(prob)
        st.plotly_chart(gauge_fig, width="stretch", key="gauge_main")

    st.markdown("#### Clinical Snapshot")
    snap_cols = st.columns(4)
    snapshot_items = [
        ("Glucose", f"{glucose} mg/dL", glucose),
        ("BMI", f"{bmi:.1f} kg/m²", bmi),
        ("Blood Pressure", f"{blood_pressure} mm Hg", blood_pressure),
        ("Age", f"{age} years", age),
    ]
    status_lookup = {
        "Blood Pressure": "BloodPressure",
    }
    for col, (title, value_text, value) in zip(snap_cols, snapshot_items):
        status_feature = status_lookup.get(title, title)
        status, colour = _clinical_status(status_feature, value)
        with col:
            st.markdown(_snapshot_card(title, value_text, status, colour), unsafe_allow_html=True)

    profile_col, actions_col = st.columns([1.05, 0.95], gap="large")
    with profile_col:
        st.markdown("#### Patient Profile Radar")
        st.plotly_chart(_profile_radar(raw_input), width="stretch", key="patient_profile_radar")

    with actions_col:
        st.markdown("#### Professional Review Checklist")
        action_items = _next_steps(raw_input, risk_label)
        action_html = "".join(f"<li>{item}</li>" for item in action_items)
        st.markdown(
            f"""
            <div class="action-card">
                <div class="snapshot-title">Suggested review points</div>
                <ul style="margin:.2rem 0 0 1rem; padding:0; color:#122033; line-height:1.65;">
                    {action_html}
                </ul>
                <p class="risk-subtext" style="margin-top:.75rem;">
                    These are educational review prompts, not medical instructions.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── SHAP Explanations ────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<span class="section-chip">Explainability layer</span>', unsafe_allow_html=True)

    try:
        shap_explainer = _get_explainer()
        from explainer import (
            get_shap_values,
            plot_waterfall,
            plot_force_html,
        )

        sv = get_shap_values(shap_explainer, X_scaled)
        shap_dict = {feat: float(v) for feat, v in zip(feature_names, sv)}

        tab_waterfall, tab_force = st.tabs(["📉 Waterfall Plot", "⚡ Force Plot"])

        with tab_waterfall:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                wf_fig = plot_waterfall(shap_explainer, X_scaled, feature_names)
            st.pyplot(wf_fig, width="stretch")
            st.caption(
                "Each bar shows how much a feature pushes the model's output toward (red) "
                "or away from (blue) a diabetes prediction."
            )

        with tab_force:
            try:
                force_html = plot_force_html(shap_explainer, X_scaled, feature_names)
                components.html(force_html, height=200, scrolling=True)
            except Exception:
                st.info("Force plot requires an internet connection to load SHAP JS library.")

        st.markdown("#### Top Model Drivers")
        top_drivers = sorted(shap_dict.items(), key=lambda item: abs(item[1]), reverse=True)[:4]
        driver_cols = st.columns(4)
        for col, (feature, contribution) in zip(driver_cols, top_drivers):
            with col:
                st.markdown(_driver_card(feature, contribution, raw_input), unsafe_allow_html=True)

    except Exception:
        st.warning(
            f"⚠️ SHAP computation encountered an issue:\n```\n{traceback.format_exc()}\n```"
        )
        shap_dict = {}
        wf_fig = None

    # ── AI Summary ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🤖 AI-Generated Patient Summary")

    @st.cache_data(show_spinner=False)
    def _get_local_summary(pd_frozen: str, sv_frozen: str, rl: str, pr: float) -> str:
        """Cache wrapper for the local text generator."""
        import sys
        sys.path.insert(0, str(BASE_DIR))
        from llm_summary import generate_summary
        pd_ = json.loads(pd_frozen)
        sv_ = json.loads(sv_frozen)
        return generate_summary(pd_, sv_, rl, pr)

    with st.spinner("Generating plain-English summary…"):
        summary = _get_local_summary(
            json.dumps(raw_input),
            json.dumps(shap_dict),
            risk_label,
            prob,
        )

    st.markdown(
        f"""
        <div class="summary-box">
            <div class="snapshot-title">Patient-friendly explanation</div>
            {summary}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Append to history ────────────────────────────────────────────
    st.session_state["history"].append(
        {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "probability": prob,
            "risk_level": risk_label,
        }
    )

    # ── PDF Report ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📄 Download Report")

    try:
        import sys
        sys.path.insert(0, str(BASE_DIR))
        from report import generate_pdf

        pdf_bytes = generate_pdf(
            patient_data=raw_input,
            risk_level=risk_label,
            probability=prob,
            llm_summary=summary,
            shap_fig=wf_fig if "wf_fig" in dir() else None,
        )

        st.download_button(
            label="📥 Download PDF Report",
            data=pdf_bytes,
            file_name="diabetes_risk_report.pdf",
            mime="application/pdf",
            width="content",
        )
    except Exception as exc:
        st.warning(f"⚠️ Could not generate PDF: {exc}")

    # Store scaled input for What-If tool
    st.session_state["last_X_scaled"] = X_scaled
    st.session_state["last_raw_input"] = raw_input
    st.session_state["last_prob"] = prob

# ─────────────────────────────────────────────
# What-If Counterfactual Tool
# ─────────────────────────────────────────────
st.markdown("---")
with st.expander("🔬 What-If Counterfactual Tool", expanded=False):
    st.markdown(
        "Adjust the sliders below to see how changing key features affects your risk prediction."
    )

    if X_train_bg is not None and "last_raw_input" in st.session_state:
        try:
            from explainer import get_global_shap_values, top_n_features

            shap_explainer = _get_explainer()
            global_sv, _ = get_global_shap_values(shap_explainer, X_train_bg, n_samples=200)
            top_feats = top_n_features(global_sv, feature_names, n=4)
            # Only keep top features that are in raw input (not engineered)
            raw_top = [f for f in top_feats if f in FEATURE_RANGES][:4]
            if not raw_top:
                raw_top = ["Glucose", "BMI", "Age", "Insulin"]

            base_input = dict(st.session_state["last_raw_input"])
            base_prob = st.session_state.get("last_prob", 0.5)

            cf_cols = st.columns(len(raw_top))
            cf_values: dict = {}
            for i, feat in enumerate(raw_top):
                mn, mx, dv = FEATURE_RANGES[feat]
                with cf_cols[i]:
                    cf_values[feat] = st.slider(
                        feat,
                        min_value=float(mn),
                        max_value=float(mx),
                        value=float(base_input.get(feat, dv)),
                        key=f"cf_{feat}",
                    )

            # Re-run prediction with counterfactual values
            cf_input = {**base_input, **cf_values}
            cf_df = pd.DataFrame([cf_input])
            cf_X = preprocess_with_median(cf_df, medians, scaler)
            cf_prob = ensemble.predict_proba(cf_X)[0, 1]
            cf_label, cf_colour = get_risk_tier(cf_prob)

            st.markdown("---")
            diff = cf_prob - base_prob
            direction = "🔺 increases" if diff > 0 else "🔻 reduces"
            for feat, new_val in cf_values.items():
                orig_val = base_input.get(feat, new_val)
                if abs(new_val - orig_val) > 0.001:
                    st.markdown(
                        f"Changing **{feat}** from `{orig_val:.1f}` to `{new_val:.1f}` "
                        f"{direction} your risk from **{base_prob:.0%}** to **{cf_prob:.0%}**."
                    )

            cf_gauge = _professional_gauge(cf_prob)
            st.plotly_chart(cf_gauge, width="stretch", key="cf_gauge")

        except Exception:
            st.info("Run a prediction first to enable the What-If tool.")
    else:
        st.info("⬆️ Run a prediction above to unlock the What-If tool.")

# ─────────────────────────────────────────────
# Risk History (Longitudinal Trend)
# ─────────────────────────────────────────────
st.markdown("---")
with st.expander("📈 Your Risk History", expanded=len(st.session_state["history"]) >= 2):
    history = st.session_state["history"]

    if len(history) < 2:
        st.info("Make at least 2 predictions to see your longitudinal trend.")
    else:
        hist_df = pd.DataFrame(history)
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(
            x=hist_df["timestamp"],
            y=hist_df["probability"] * 100,
            mode="lines+markers+text",
            text=[f"{p:.0%}" for p in hist_df["probability"]],
            textposition="top center",
            line=dict(color="#818cf8", width=2.5),
            marker=dict(size=10, color="#a78bfa"),
            fill="tozeroy",
            fillcolor="rgba(129,140,248,0.1)",
            name="Risk %",
        ))
        fig_hist.add_hline(y=35, line_dash="dot", line_color="#22c55e",
                           annotation_text="Low Risk Boundary", opacity=0.7)
        fig_hist.add_hline(y=65, line_dash="dot", line_color="#ef4444",
                           annotation_text="High Risk Boundary", opacity=0.7)
        fig_hist.update_layout(
            title="Diabetes Risk Over Time",
            xaxis_title="Assessment Time",
            yaxis_title="Risk Probability (%)",
            yaxis=dict(range=[0, 105]),
            paper_bgcolor="#0f172a",
            plot_bgcolor="#1e293b",
            font=dict(color="#f1f5f9"),
            height=320,
        )
        st.plotly_chart(fig_hist, width="stretch", key="history_chart")

    if history:
        if st.button("🗑️ Clear History"):
            st.session_state["history"] = []
            st.rerun()
