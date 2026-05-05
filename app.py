"""
Main Streamlit dashboard for the Diabetes Risk XAI project.

Run with:
    python3 -m streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ui import apply_theme, metric_card, readiness_card
from utils import (
    ALL_FEATURES,
    DATA_DIR,
    FEATURE_RANGES,
    RAW_FEATURE_COLS,
    ZERO_IMPUTE_COLS,
    get_risk_tier,
    load_models,
    preprocess_with_median,
)


st.set_page_config(
    page_title="Diabetes Risk XAI",
    page_icon="D",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
METRICS_PATH = MODELS_DIR / "metrics.json"
MEDIANS_PATH = MODELS_DIR / "medians.json"

MODEL_ARTIFACTS = {
    "Ensemble model": MODELS_DIR / "ensemble_model.pkl",
    "Explainability model": MODELS_DIR / "rf_model.pkl",
    "Feature scaler": MODELS_DIR / "scaler.pkl",
    "Feature names": MODELS_DIR / "feature_names.pkl",
    "Training background": MODELS_DIR / "training_data.pkl",
    "Training medians": MEDIANS_PATH,
    "Metrics": METRICS_PATH,
}


@st.cache_data
def load_metrics() -> dict:
    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            return json.load(f)
    return {}


@st.cache_data
def load_medians() -> dict:
    if MEDIANS_PATH.exists():
        with open(MEDIANS_PATH) as f:
            return json.load(f)
    return {}


@st.cache_resource
def load_runtime():
    return load_models()


@st.cache_data
def load_dataset() -> pd.DataFrame | None:
    csv_path = DATA_DIR / "diabetes.csv"
    if not csv_path.exists():
        return None

    from model import COLUMN_NAMES

    df = pd.read_csv(csv_path, header=None, names=COLUMN_NAMES)
    clean = df.copy()
    for col in ZERO_IMPUTE_COLS:
        clean[col] = clean[col].replace(0, np.nan)
        clean[col] = clean[col].fillna(clean[col].median())
    clean["GlucoseBMI"] = clean["Glucose"] * clean["BMI"] / 100.0
    clean["AgeRisk"] = clean["Age"] * clean["DiabetesPedigreeFunction"]
    clean["InsulinResistance"] = clean["Insulin"] / (clean["BMI"] + 1.0)
    return clean


def home_gauge(probability: float) -> go.Figure:
    label, color = get_risk_tier(probability)
    pct = probability * 100
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pct,
            number={"suffix": "%", "font": {"size": 34, "color": "#132238"}},
            title={
                "text": f"<b>{label}</b>",
                "font": {"size": 18, "color": color},
            },
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#5d6f86"},
                "bar": {"color": color, "thickness": 0.28},
                "bgcolor": "#ffffff",
                "borderwidth": 1,
                "bordercolor": "#d9e3ee",
                "steps": [
                    {"range": [0, 35], "color": "rgba(22, 163, 74, 0.16)"},
                    {"range": [35, 65], "color": "rgba(217, 119, 6, 0.18)"},
                    {"range": [65, 100], "color": "rgba(220, 38, 38, 0.16)"},
                ],
                "threshold": {
                    "line": {"color": "#132238", "width": 3},
                    "thickness": 0.78,
                    "value": pct,
                },
            },
        )
    )
    fig.update_layout(
        height=290,
        margin=dict(t=45, b=0, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def predict_patient(patient: dict, ensemble, scaler, medians: dict) -> float:
    df = pd.DataFrame([patient])
    x_scaled = preprocess_with_median(df, medians, scaler)
    return float(ensemble.predict_proba(x_scaled)[0, 1])


def sensitivity_table(patient: dict, base_prob: float, ensemble, scaler, medians: dict) -> pd.DataFrame:
    rows = []
    for feature in RAW_FEATURE_COLS:
        mn, mx, default = FEATURE_RANGES[feature]
        probe_low = dict(patient)
        probe_high = dict(patient)
        probe_low[feature] = mn
        probe_high[feature] = mx
        low_prob = predict_patient(probe_low, ensemble, scaler, medians)
        high_prob = predict_patient(probe_high, ensemble, scaler, medians)
        strongest = max(abs(low_prob - base_prob), abs(high_prob - base_prob))
        direction = "Higher value raises risk" if high_prob > low_prob else "Lower value raises risk"
        rows.append(
            {
                "Feature": feature,
                "Current": patient[feature],
                "Low scenario": round(low_prob * 100, 1),
                "High scenario": round(high_prob * 100, 1),
                "Sensitivity": round(strongest * 100, 1),
                "Pattern": direction,
            }
        )
    return pd.DataFrame(rows).sort_values("Sensitivity", ascending=False)


def format_status(path: Path) -> str:
    return "Ready" if path.exists() else "Missing"


metrics = load_metrics()
medians = load_medians()
dataset = load_dataset()

try:
    ensemble, rf_model, scaler, feature_names = load_runtime()
    runtime_ready = True
except FileNotFoundError:
    ensemble = rf_model = scaler = feature_names = None
    runtime_ready = False

with st.sidebar:
    st.markdown("## Diabetes Risk XAI")
    st.markdown("Clinical machine-learning dashboard with explainability, batch analysis, and fairness audit.")
    st.markdown("---")

    ready_count = sum(path.exists() for path in MODEL_ARTIFACTS.values())
    st.markdown(f"**System readiness:** {ready_count}/{len(MODEL_ARTIFACTS)} artifacts")
    st.progress(ready_count / len(MODEL_ARTIFACTS))

    if runtime_ready:
        st.success("Prediction engine ready")
    else:
        st.error("Model artifacts missing")

    st.markdown("---")
    st.caption("Research prototype only. It is not a medical diagnosis or treatment recommendation.")


st.markdown(
    """
    <div class="dashboard-band">
        <div class="section-kicker">Major Project Dashboard</div>
        <h1 style="margin:0 0 .25rem 0;">Diabetes Risk XAI</h1>
        <p class="small-note" style="margin:0; max-width:780px;">
            A complete explainable-AI workflow for diabetes risk screening: live prediction,
            individual explanations, batch scoring, global model insights, report generation,
            and fairness validation.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

metric_cols = st.columns(4)
metric_values = [
    ("Accuracy", f"{metrics.get('accuracy', 0):.1%}" if metrics else "-"),
    ("ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}" if metrics else "-"),
    ("F1 Score", f"{metrics.get('f1', 0):.3f}" if metrics else "-"),
    ("Dataset Rows", str(int(metrics.get("dataset_size", 0))) if metrics else "-"),
]
metric_colors = ["#2563eb", "#0f9f8f", "#d97706", "#16a34a"]
for col, (label, value), color in zip(metric_cols, metric_values, metric_colors):
    with col:
        st.markdown(metric_card(label, value, color), unsafe_allow_html=True)

if not runtime_ready:
    st.error("Model files are missing. Run `python3 model.py` from the project folder, then restart Streamlit.")
    st.stop()

tab_assess, tab_model, tab_data, tab_ready = st.tabs(
    ["Rapid Assessment", "Model Intelligence", "Dataset Explorer", "Project Readiness"]
)

with tab_assess:
    presets = {
        "Balanced baseline": {
            "Pregnancies": 3,
            "Glucose": 117,
            "BloodPressure": 72,
            "SkinThickness": 29,
            "Insulin": 125,
            "BMI": 32.0,
            "DiabetesPedigreeFunction": 0.47,
            "Age": 33,
        },
        "Higher-risk pattern": {
            "Pregnancies": 6,
            "Glucose": 168,
            "BloodPressure": 82,
            "SkinThickness": 36,
            "Insulin": 210,
            "BMI": 39.5,
            "DiabetesPedigreeFunction": 0.92,
            "Age": 49,
        },
        "Lower-risk pattern": {
            "Pregnancies": 1,
            "Glucose": 92,
            "BloodPressure": 68,
            "SkinThickness": 22,
            "Insulin": 80,
            "BMI": 24.6,
            "DiabetesPedigreeFunction": 0.22,
            "Age": 26,
        },
    }

    scenario = st.selectbox("Clinical scenario", list(presets.keys()), index=0)
    defaults = presets[scenario]

    input_col, result_col = st.columns([1.08, 0.92], gap="large")

    with input_col:
        st.subheader("Patient Measurements")
        c1, c2 = st.columns(2)
        patient: dict = {}
        for idx, feature in enumerate(RAW_FEATURE_COLS):
            mn, mx, default = FEATURE_RANGES[feature]
            value = defaults.get(feature, default)
            target = c1 if idx < 4 else c2
            with target:
                if isinstance(default, float):
                    patient[feature] = st.slider(
                        feature,
                        min_value=float(mn),
                        max_value=float(mx),
                        value=float(value),
                        step=0.01 if feature == "DiabetesPedigreeFunction" else 0.1,
                    )
                else:
                    patient[feature] = st.slider(
                        feature,
                        min_value=int(mn),
                        max_value=int(mx),
                        value=int(value),
                        step=1,
                    )

    probability = predict_patient(patient, ensemble, scaler, medians)
    risk_label, risk_color = get_risk_tier(probability)
    sensitivity = sensitivity_table(patient, probability, ensemble, scaler, medians)

    with result_col:
        st.subheader("Live Risk Score")
        st.plotly_chart(home_gauge(probability), width="stretch", key="home_gauge")
        st.markdown(
            f"""
            <div class="risk-panel">
                <div class="risk-title">Current prediction</div>
                <div class="risk-value" style="color:{risk_color}">{risk_label} - {probability:.1%}</div>
                <div class="small-note">Open Single Patient for SHAP waterfall, force plot, PDF report, and counterfactual history.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.subheader("Sensitivity Scan")
    scan_col, table_col = st.columns([0.9, 1.1], gap="large")

    with scan_col:
        top_scan = sensitivity.head(6).sort_values("Sensitivity")
        fig = px.bar(
            top_scan,
            x="Sensitivity",
            y="Feature",
            orientation="h",
            color="Sensitivity",
            color_continuous_scale=["#b7e4dc", "#2563eb", "#d97706"],
            labels={"Sensitivity": "Max probability change (%)"},
        )
        fig.update_layout(
            height=350,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#ffffff",
            font=dict(color="#132238"),
            coloraxis_showscale=False,
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(fig, width="stretch")

    with table_col:
        st.dataframe(
            sensitivity,
            width="stretch",
            hide_index=True,
            column_config={
                "Low scenario": st.column_config.NumberColumn("Low scenario risk %", format="%.1f"),
                "High scenario": st.column_config.NumberColumn("High scenario risk %", format="%.1f"),
                "Sensitivity": st.column_config.ProgressColumn(
                    "Sensitivity %",
                    min_value=0,
                    max_value=max(1.0, float(sensitivity["Sensitivity"].max())),
                    format="%.1f",
                ),
            },
        )

with tab_model:
    st.subheader("Model Architecture")
    c1, c2, c3, c4 = st.columns(4)
    cards = [
        ("Random Forest", "nonlinear baseline", "#2563eb"),
        ("Gradient Boosting", "boosted trees", "#0f9f8f"),
        ("Histogram GB", "fast boosted trees", "#d97706"),
        ("Logistic Regression", "linear calibration", "#16a34a"),
    ]
    for col, (title, caption, color) in zip([c1, c2, c3, c4], cards):
        with col:
            st.markdown(
                f"<div class='workflow-card'><div class='metric-value' style='font-size:1.05rem;color:{color}'>{title}</div>"
                f"<div class='small-note'>{caption}</div></div>",
                unsafe_allow_html=True,
            )

    st.markdown("---")
    metric_df = pd.DataFrame(
        {
            "Metric": ["Accuracy", "ROC-AUC", "F1", "Precision", "Recall"],
            "Score": [
                metrics.get("accuracy", np.nan),
                metrics.get("roc_auc", np.nan),
                metrics.get("f1", np.nan),
                metrics.get("precision", np.nan),
                metrics.get("recall", np.nan),
            ],
        }
    ).dropna()

    if not metric_df.empty:
        fig = px.bar(
            metric_df,
            x="Metric",
            y="Score",
            color="Metric",
            color_discrete_sequence=["#2563eb", "#0f9f8f", "#d97706", "#16a34a", "#dc2626"],
            text=metric_df["Score"].map(lambda x: f"{x:.3f}"),
        )
        fig.update_layout(
            yaxis=dict(range=[0, 1]),
            height=360,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#ffffff",
            font=dict(color="#132238"),
            showlegend=False,
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(fig, width="stretch")

    st.info(
        "The ensemble is trained on engineered clinical features and uses a standalone Random Forest for stable SHAP explanations."
    )

with tab_data:
    if dataset is None:
        st.warning("Dataset not found. Run `python3 model.py` to download and prepare it.")
    else:
        st.subheader("Training Dataset Profile")
        d1, d2, d3 = st.columns(3)
        with d1:
            positive_rate = dataset["Outcome"].mean()
            st.markdown(metric_card("Positive Class Rate", f"{positive_rate:.1%}", "#dc2626"), unsafe_allow_html=True)
        with d2:
            st.markdown(metric_card("Median Glucose", f"{dataset['Glucose'].median():.0f}", "#2563eb"), unsafe_allow_html=True)
        with d3:
            st.markdown(metric_card("Median BMI", f"{dataset['BMI'].median():.1f}", "#0f9f8f"), unsafe_allow_html=True)

        chart1, chart2 = st.columns(2, gap="large")
        with chart1:
            scatter = px.scatter(
                dataset,
                x="Glucose",
                y="BMI",
                color=dataset["Outcome"].map({0: "No Diabetes", 1: "Diabetes"}),
                color_discrete_map={"No Diabetes": "#2563eb", "Diabetes": "#dc2626"},
                hover_data=["Age", "Pregnancies", "DiabetesPedigreeFunction"],
                labels={"color": "Outcome"},
                title="Glucose vs BMI by Outcome",
            )
            scatter.update_layout(
                height=420,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#ffffff",
                font=dict(color="#132238"),
            )
            st.plotly_chart(scatter, width="stretch")

        with chart2:
            corr = dataset[ALL_FEATURES + ["Outcome"]].corr()
            heat = px.imshow(
                corr,
                text_auto=".2f",
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                title="Feature Correlation Matrix",
                aspect="auto",
            )
            heat.update_layout(
                height=420,
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#132238"),
            )
            st.plotly_chart(heat, width="stretch")

        st.markdown("---")
        st.dataframe(dataset[RAW_FEATURE_COLS + ["Outcome"]].describe().round(2), width="stretch")

with tab_ready:
    st.subheader("Project Readiness")
    ready_cols = st.columns(3)
    with ready_cols[0]:
        st.markdown(readiness_card("Model Artifacts", f"{ready_count}/{len(MODEL_ARTIFACTS)}", "#0f9f8f"), unsafe_allow_html=True)
    with ready_cols[1]:
        st.markdown(readiness_card("Pages", "5", "#2563eb"), unsafe_allow_html=True)
    with ready_cols[2]:
        st.markdown(readiness_card("Exports", "CSV + PDF", "#d97706"), unsafe_allow_html=True)

    artifact_df = pd.DataFrame(
        [{"Component": name, "Path": str(path.relative_to(BASE_DIR)), "Status": format_status(path)} for name, path in MODEL_ARTIFACTS.items()]
    )
    st.dataframe(
        artifact_df,
        width="stretch",
        hide_index=True,
        column_config={
            "Status": st.column_config.SelectboxColumn(
                "Status",
                options=["Ready", "Missing"],
            )
        },
    )

    st.markdown("### Run Commands")
    st.code(
        "cd '/Users/jabirimteyaz/Desktop/Major Project/diabetes_xai'\n"
        "python3 -m pip install -r requirements.txt\n"
        "python3 model.py\n"
        "python3 -m streamlit run app.py",
        language="bash",
    )
