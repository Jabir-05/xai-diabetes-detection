"""
llm_summary.py — Plain-English risk summaries via Anthropic Claude API.

Reads ANTHROPIC_API_KEY from environment or .env file.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root
load_dotenv(Path(__file__).parent / ".env")


def _build_prompt(
    patient_data: dict,
    shap_values: dict,
    risk_level: str,
    probability: float,
) -> str:
    """
    Construct the Claude prompt from patient data and SHAP values.

    Parameters
    ----------
    patient_data : dict
        Feature name → raw (un-scaled) patient value.
    shap_values : dict
        Feature name → SHAP value (positive = increases risk).
    risk_level : str
        "Low Risk", "Borderline Risk", or "High Risk".
    probability : float
        Predicted probability of diabetes (0–1).

    Returns
    -------
    str
        Formatted prompt string.
    """
    shap_lines = "\n".join(
        f"  • {feat}: {val:+.4f}"
        for feat, val in sorted(shap_values.items(), key=lambda x: -abs(x[1]))
    )
    patient_lines = "\n".join(
        f"  • {feat}: {val}" for feat, val in patient_data.items()
    )

    return (
        f"You are a clinical assistant. A patient has been assessed for diabetes risk.\n"
        f"Risk level: {risk_level} ({probability:.0%} probability).\n\n"
        f"Top contributing factors (SHAP values, positive = increases risk):\n"
        f"{shap_lines}\n\n"
        f"Patient values:\n"
        f"{patient_lines}\n\n"
        f"Write a clear, empathetic 3-sentence explanation suitable for the patient "
        f"(not a clinician). Do not use medical jargon. End with one actionable tip."
    )


def generate_summary(
    patient_data: dict,
    shap_values: dict,
    risk_level: str,
    probability: float,
) -> str | None:
    """
    Generate a plain-English risk summary using local rule-based logic based on SHAP values.
    Does not require an API key.

    Parameters
    ----------
    patient_data : dict
        Feature name → raw patient value.
    shap_values : dict
        Feature name → SHAP value.
    risk_level : str
        Risk tier label.
    probability : float
        Predicted probability (0–1).

    Returns
    -------
    str
        The generated plain-English text.
    """
    # Sort features by SHAP magnitude
    sorted_feats = sorted(shap_values.items(), key=lambda x: -abs(x[1]))
    
    # Separate into risk-increasing and risk-decreasing
    increasing = [f for f, v in sorted_feats if v > 0]
    decreasing = [f for f, v in sorted_feats if v < 0]

    # Build sentence 1: Overall risk
    s1 = f"Based on your profile, your diabetes risk assessment is **{risk_level}** with a {probability:.0%} probability."
    
    # Build sentence 2: Driving factors
    if increasing and len(increasing) >= 2:
        s2 = f"The primary factors increasing this risk are your {increasing[0]} ({patient_data.get(increasing[0], 'N/A')}) and {increasing[1]} ({patient_data.get(increasing[1], 'N/A')})."
    elif increasing:
        s2 = f"The primary factor increasing this risk is your {increasing[0]} ({patient_data.get(increasing[0], 'N/A')})."
    else:
        s2 = "There are no major individual factors heavily driving up your risk."

    # Build sentence 3: Protective factors or general tip
    if decreasing:
        s3 = f"However, your {decreasing[0]} ({patient_data.get(decreasing[0], 'N/A')}) is currently helping to lower your overall risk."
    else:
        s3 = "Focus on maintaining a healthy lifestyle to keep your risk levels managed."

    # Build sentence 4: Tip
    if "Glucose" in increasing[:2] or "GlucoseBMI" in increasing[:2]:
        s4 = "Consider discussing dietary changes to manage your blood sugar with your doctor."
    elif "BMI" in increasing[:2]:
        s4 = "A tailored exercise and diet plan could help improve your BMI and lower your risk."
    else:
        s4 = "We recommend sharing these results with your healthcare provider for a detailed clinical review."

    return f"{s1} {s2} {s3} {s4}"
