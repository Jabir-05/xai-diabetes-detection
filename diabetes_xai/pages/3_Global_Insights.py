"""
pages/3_Global_Insights.py — Global model explainability: SHAP beeswarm, bar plot,
feature correlation heatmap, and class distribution.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = Path(__file__).parent.parent

st.set_page_config(page_title="Global Insights | Diabetes XAI", page_icon="🌍", layout="wide")

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
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🌍 Global Insights")
    st.markdown(
        "Explore how features collectively influence the model's decisions "
        "across the entire training dataset."
    )
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
from utils import ALL_FEATURES, DATA_DIR


@st.cache_resource
def _load():
    """Load model artefacts once per session."""
    from utils import load_models
    return load_models()


@st.cache_resource
def _load_training_data():
    """Load raw training arrays for SHAP background."""
    import joblib
    path = BASE_DIR / "models" / "training_data.pkl"
    if path.exists():
        return joblib.load(path)
    return None, None


@st.cache_resource
def _get_explainer(_rf_model, _X_bg):
    """Build and cache the SHAP TreeExplainer."""
    from explainer import get_explainer
    return get_explainer(_rf_model, _X_bg)


@st.cache_data
def _compute_global_shap(n_samples: int = 300):
    """Compute global SHAP values (cached, runs once per session)."""
    from explainer import get_global_shap_values
    shap_matrix, X_subset = get_global_shap_values(explainer, X_train_bg, n_samples=n_samples)
    return shap_matrix, X_subset


@st.cache_data
def _load_raw_dataset() -> pd.DataFrame | None:
    """Load the raw PIMA dataset for correlation analysis."""
    csv_path = DATA_DIR / "diabetes.csv"
    if csv_path.exists():
        from model import COLUMN_NAMES
        try:
            df = pd.read_csv(csv_path, header=None, names=COLUMN_NAMES)
            return df
        except Exception:
            return None
    return None


try:
    ensemble, rf_model, scaler, feature_names = _load()
    X_train_bg, y_train_bg = _load_training_data()
    models_ok = True
except FileNotFoundError as e:
    st.error(f"❌ {e}")
    models_ok = False

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='background:linear-gradient(135deg,#818cf8,#a78bfa);-webkit-background-clip:text;"
    "-webkit-text-fill-color:transparent;background-clip:text;'>🌍 Global Model Insights</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "Understand which features drive predictions across the entire dataset, "
    "using global SHAP analysis and exploratory visualisations."
)

if not models_ok or X_train_bg is None:
    st.warning("⚠️ Training data not found. Re-run `python model.py` to regenerate artefacts.")
    st.stop()

# ── Build explainer + global SHAP ────────────────────────────────────────────
explainer = _get_explainer(rf_model, X_train_bg)

with st.spinner("Computing global SHAP values (this may take 10–20 seconds)…"):
    shap_matrix, X_subset = _compute_global_shap(n_samples=250)

# ─────────────────────────────────────────────────────────────────────────────
# Tab layout
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🐝 Beeswarm Plot",
    "📊 Feature Importance Bar",
    "🔥 Correlation Heatmap",
    "🥧 Class Distribution",
])

# ── Tab 1: Beeswarm ───────────────────────────────────────────────────────────
with tab1:
    st.markdown("#### SHAP Beeswarm Plot")
    st.markdown(
        "Each dot is one training sample. The horizontal position shows the SHAP value "
        "(impact on prediction), and colour indicates the feature value (red = high, blue = low)."
    )
    try:
        from explainer import plot_beeswarm
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bee_fig = plot_beeswarm(shap_matrix, X_subset, feature_names, max_display=11)
        st.pyplot(bee_fig, width="stretch")
    except Exception as exc:
        st.warning(f"⚠️ Could not render beeswarm: {exc}")

# ── Tab 2: Bar Plot ───────────────────────────────────────────────────────────
with tab2:
    st.markdown("#### Mean |SHAP Value| — Global Feature Importance")
    st.markdown(
        "The bar length shows the average magnitude of each feature's impact on the model's output. "
        "Longer bars = more influential features."
    )
    try:
        from explainer import plot_shap_bar
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bar_fig = plot_shap_bar(shap_matrix, feature_names, max_display=11)
        st.pyplot(bar_fig, width="stretch")
    except Exception as exc:
        st.warning(f"⚠️ Could not render bar plot: {exc}")

    # Also show interactive Plotly version
    st.markdown("---")
    st.markdown("##### Interactive Feature Importance (Plotly)")
    mean_abs = np.abs(shap_matrix).mean(axis=0)
    sorted_idx = np.argsort(mean_abs)[::-1]
    feat_imp_df = pd.DataFrame({
        "Feature": [feature_names[i] for i in sorted_idx],
        "Mean |SHAP|": mean_abs[sorted_idx],
    })

    plotly_bar = px.bar(
        feat_imp_df,
        x="Mean |SHAP|",
        y="Feature",
        orientation="h",
        color="Mean |SHAP|",
        color_continuous_scale="Plasma",
        title="Global Feature Importance (Mean |SHAP Value|)",
    )
    plotly_bar.update_layout(
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        font=dict(color="#f1f5f9"),
        yaxis=dict(autorange="reversed"),
        coloraxis_showscale=False,
        height=420,
    )
    st.plotly_chart(plotly_bar, width="stretch")

# ── Tab 3: Correlation Heatmap ────────────────────────────────────────────────
with tab3:
    st.markdown("#### Feature Correlation Heatmap")
    st.markdown(
        "Pearson correlations between all features in the training set. "
        "Strong correlations (near ±1) may indicate redundancy."
    )
    try:
        df_raw = _load_raw_dataset()
        if df_raw is not None:
            # Compute engineered features for full correlation picture
            df_eng = df_raw.copy()
            df_eng["GlucoseBMI"] = df_eng["Glucose"] * df_eng["BMI"] / 100.0
            df_eng["AgeRisk"] = df_eng["Age"] * df_eng["DiabetesPedigreeFunction"]
            df_eng["InsulinResistance"] = df_eng["Insulin"] / (df_eng["BMI"] + 1.0)

            corr = df_eng[ALL_FEATURES + ["Outcome"]].corr()

            heat_fig = px.imshow(
                corr,
                text_auto=".2f",
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                title="Feature Correlation Matrix",
                aspect="auto",
            )
            heat_fig.update_layout(
                paper_bgcolor="#0f172a",
                font=dict(color="#f1f5f9"),
                height=540,
            )
            heat_fig.update_coloraxes(colorbar_tickfont=dict(color="#f1f5f9"))
            st.plotly_chart(heat_fig, width="stretch")
        else:
            st.info("Raw dataset not found. Run `python model.py` to download it.")
    except Exception as exc:
        st.warning(f"⚠️ Could not render heatmap: {exc}")

# ── Tab 4: Class Distribution ─────────────────────────────────────────────────
with tab4:
    st.markdown("#### Dataset Class Distribution")
    try:
        df_raw = _load_raw_dataset()
        if df_raw is not None:
            counts = df_raw["Outcome"].value_counts().sort_index()
            labels = ["No Diabetes (0)", "Diabetes (1)"]
            values = [counts.get(0, 0), counts.get(1, 0)]

            cl1, cl2 = st.columns(2)
            with cl1:
                pie_fig = go.Figure(go.Pie(
                    labels=labels,
                    values=values,
                    marker_colors=["#22c55e", "#ef4444"],
                    hole=0.5,
                    textinfo="percent+label",
                    textfont_size=13,
                ))
                pie_fig.update_layout(
                    title="Class Balance",
                    paper_bgcolor="#0f172a",
                    font=dict(color="#f1f5f9"),
                    height=360,
                )
                st.plotly_chart(pie_fig, width="stretch")

            with cl2:
                # Feature distributions by outcome
                feature_to_plot = st.selectbox(
                    "Select feature to compare by outcome:",
                    options=ALL_FEATURES,
                    index=1,  # Glucose by default
                )
                box_fig = px.box(
                    df_raw if feature_to_plot in df_raw.columns else (
                        df_raw.assign(
                            GlucoseBMI=df_raw["Glucose"] * df_raw["BMI"] / 100,
                            AgeRisk=df_raw["Age"] * df_raw["DiabetesPedigreeFunction"],
                            InsulinResistance=df_raw["Insulin"] / (df_raw["BMI"] + 1),
                        )
                    ),
                    x="Outcome",
                    y=feature_to_plot,
                    color="Outcome",
                    color_discrete_map={0: "#22c55e", 1: "#ef4444"},
                    labels={"Outcome": "Diabetes"},
                    title=f"{feature_to_plot} by Diabetes Status",
                )
                box_fig.update_layout(
                    paper_bgcolor="#0f172a",
                    plot_bgcolor="#1e293b",
                    font=dict(color="#f1f5f9"),
                    showlegend=False,
                    height=360,
                )
                st.plotly_chart(box_fig, width="stretch")

            # Summary stats table
            st.markdown("---")
            st.markdown("##### Descriptive Statistics by Outcome")
            df_stats = df_raw.groupby("Outcome")[
                ["Glucose", "BMI", "Age", "Insulin", "BloodPressure"]
            ].mean().round(2)
            df_stats.index = ["No Diabetes", "Diabetes"]
            st.dataframe(df_stats, width="stretch")
        else:
            st.info("Raw dataset not found. Run `python model.py` to download it.")
    except Exception as exc:
        st.warning(f"⚠️ Could not render class distribution: {exc}")
