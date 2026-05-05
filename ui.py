"""
Shared Streamlit UI helpers for the Diabetes Risk XAI dashboard.
"""

from __future__ import annotations

import streamlit as st


THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --surface: #ffffff;
    --surface-soft: #f5f8fb;
    --ink: #132238;
    --muted: #5d6f86;
    --line: #d9e3ee;
    --blue: #2563eb;
    --teal: #0f9f8f;
    --green: #16a34a;
    --amber: #d97706;
    --red: #dc2626;
    --navy: #10243c;
}

html, body, [class*="css"] {
    font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 12% 0%, rgba(37, 99, 235, 0.10), transparent 30rem),
        linear-gradient(180deg, #f8fbff 0%, #eef4f9 100%);
    color: var(--ink);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #10243c 0%, #173a52 100%);
    border-right: 1px solid rgba(255, 255, 255, 0.14);
}

[data-testid="stSidebar"] * {
    color: #edf6ff !important;
}

[data-testid="stSidebar"] .stCodeBlock pre {
    background: rgba(255, 255, 255, 0.10) !important;
}

h1, h2, h3 {
    color: var(--ink) !important;
    letter-spacing: 0;
}

p, li, label, span {
    letter-spacing: 0;
}

hr {
    border-color: var(--line) !important;
}

.stButton > button,
.stDownloadButton > button,
[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(135deg, var(--blue), var(--teal)) !important;
    color: white !important;
    border: 0 !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    min-height: 2.7rem;
    box-shadow: 0 10px 24px rgba(37, 99, 235, 0.18);
}

.stButton > button:hover,
.stDownloadButton > button:hover,
[data-testid="stFormSubmitButton"] button:hover {
    box-shadow: 0 14px 30px rgba(15, 159, 143, 0.24);
    transform: translateY(-1px);
}

.dashboard-band {
    background: linear-gradient(135deg, #ffffff 0%, #edf7f6 100%);
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 1.4rem 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 14px 38px rgba(22, 46, 78, 0.08);
}

.section-kicker {
    color: var(--teal);
    font-size: 0.76rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.25rem;
}

.metric-card,
.stat-card,
.readiness-card,
.workflow-card {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 1rem;
    box-shadow: 0 8px 22px rgba(19, 34, 56, 0.06);
}

.metric-card:hover,
.stat-card:hover,
.readiness-card:hover,
.workflow-card:hover {
    border-color: rgba(15, 159, 143, 0.42);
}

.metric-value,
.stat-val,
.readiness-value {
    color: var(--ink);
    font-size: 1.75rem;
    font-weight: 800;
    line-height: 1.05;
}

.metric-label,
.stat-label,
.readiness-label {
    color: var(--muted);
    font-size: 0.74rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    margin-top: 0.35rem;
    text-transform: uppercase;
}

.risk-panel {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 1rem;
}

.risk-title {
    font-size: 0.82rem;
    color: var(--muted);
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

.risk-value {
    font-size: 2rem;
    font-weight: 800;
    margin-top: 0.25rem;
}

.pill {
    display: inline-flex;
    align-items: center;
    border-radius: 999px;
    padding: 0.25rem 0.7rem;
    background: #eaf3ff;
    color: var(--blue);
    border: 1px solid #bfdbfe;
    font-size: 0.78rem;
    font-weight: 700;
}

.small-note {
    color: var(--muted);
    font-size: 0.86rem;
}

.dataframe {
    border-radius: 8px;
}
</style>
"""


def apply_theme() -> None:
    """Apply the shared Streamlit theme."""
    st.markdown(THEME_CSS, unsafe_allow_html=True)


def metric_card(label: str, value: str, color: str = "#2563eb") -> str:
    """Return a compact HTML metric card."""
    return (
        "<div class='metric-card'>"
        f"<div class='metric-value' style='color:{color}'>{value}</div>"
        f"<div class='metric-label'>{label}</div>"
        "</div>"
    )


def readiness_card(label: str, value: str, color: str = "#0f9f8f") -> str:
    """Return a status card used on the readiness panel."""
    return (
        "<div class='readiness-card'>"
        f"<div class='readiness-value' style='color:{color}'>{value}</div>"
        f"<div class='readiness-label'>{label}</div>"
        "</div>"
    )
