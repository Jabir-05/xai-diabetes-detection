"""
report.py — PDF report generation using FPDF2.

Exposes:
    generate_pdf(patient_data, risk_level, probability,
                 llm_summary, shap_fig) -> bytes
"""

from __future__ import annotations

import io
import os
import tempfile
import unicodedata
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).parent
MPL_CACHE_DIR = Path(tempfile.gettempdir()) / "diabetes_xai_matplotlib"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib.pyplot as plt
from fpdf import FPDF


def _pdf_text(value: object) -> str:
    """Convert text to a form supported by FPDF core fonts."""
    text = "" if value is None else str(value)
    replacements = {
        "–": "-",
        "—": "-",
        "•": "-",
        "≤": "<=",
        "≥": ">=",
        "🩺": "",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = unicodedata.normalize("NFKD", text)
    return text.encode("latin-1", "ignore").decode("latin-1")


class DiabetesReportPDF(FPDF):
    """Custom FPDF subclass with styled header and footer."""

    def header(self) -> None:
        """Render the report header with title and timestamp."""
        self.set_font("Helvetica", "B", 18)
        self.set_text_color(30, 30, 60)
        self.cell(0, 12, "Diabetes Risk Assessment Report", align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 10)
        self.set_text_color(120, 120, 120)
        ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        self.cell(0, 6, _pdf_text(f"Generated: {ts}"), align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(4)
        self.set_draw_color(80, 80, 180)
        self.set_line_width(0.5)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(6)

    def footer(self) -> None:
        """Render the disclaimer footer."""
        self.set_y(-18)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(
            0,
            8,
            _pdf_text(
                "This report is for informational purposes only. "
                "It does not constitute medical advice or diagnosis. "
                "Please consult a qualified healthcare professional."
            ),
            align="C",
        )


def _risk_colour(risk_level: str) -> tuple[int, int, int]:
    """Return an RGB tuple for the risk tier."""
    mapping = {
        "Low Risk": (34, 197, 94),
        "Borderline Risk": (249, 115, 22),
        "High Risk": (239, 68, 68),
    }
    return mapping.get(risk_level, (100, 100, 100))


def generate_pdf(
    patient_data: dict,
    risk_level: str,
    probability: float,
    llm_summary: str | None,
    shap_fig: plt.Figure | None,
) -> bytes:
    """
    Generate a styled PDF diabetes risk report.

    Parameters
    ----------
    patient_data : dict
        Feature name → patient value.
    risk_level : str
        "Low Risk", "Borderline Risk", or "High Risk".
    probability : float
        Predicted probability (0–1).
    llm_summary : str | None
        Plain-English AI summary from Claude, or None.
    shap_fig : matplotlib.figure.Figure | None
        SHAP waterfall figure to embed, or None.

    Returns
    -------
    bytes
        Raw PDF bytes suitable for st.download_button().
    """
    pdf = DiabetesReportPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # ── Risk Level Banner ────────────────────────────────────────────
    r, g, b = _risk_colour(risk_level)
    pdf.set_fill_color(r, g, b)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 22)
    pdf.cell(0, 16, _pdf_text(f"{risk_level}  ({probability:.1%} probability)"), align="C",
             fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)

    # ── Patient Input Values ─────────────────────────────────────────
    pdf.set_text_color(30, 30, 60)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, _pdf_text("Patient Input Values"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    col_w = (pdf.w - pdf.l_margin - pdf.r_margin) / 2
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_fill_color(220, 220, 240)
    pdf.set_text_color(50, 50, 50)
    pdf.cell(col_w, 8, _pdf_text("Feature"), border=1, fill=True)
    pdf.cell(col_w, 8, _pdf_text("Value"), border=1, fill=True, new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 10)
    for i, (feat, val) in enumerate(patient_data.items()):
        fill = i % 2 == 0
        pdf.set_fill_color(240, 240, 255) if fill else pdf.set_fill_color(255, 255, 255)
        pdf.cell(col_w, 7, _pdf_text(feat), border=1, fill=fill)
        pdf.cell(col_w, 7, _pdf_text(round(val, 4) if isinstance(val, float) else val),
                 border=1, fill=fill, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)

    # ── AI Summary ───────────────────────────────────────────────────
    pdf.set_text_color(30, 30, 60)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, _pdf_text("AI-Generated Patient Summary"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 11)

    if llm_summary:
        pdf.set_fill_color(235, 240, 255)
        pdf.multi_cell(0, 7, _pdf_text(llm_summary), fill=True, border=0)
    else:
        pdf.set_text_color(160, 160, 160)
        pdf.cell(0, 8, _pdf_text("(AI summary not available - local summary not generated)"),
                 new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)

    # ── SHAP Waterfall Chart ─────────────────────────────────────────
    if shap_fig is not None:
        pdf.set_text_color(30, 30, 60)
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, _pdf_text("SHAP Explanation (Waterfall Chart)"), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

        # Save matplotlib figure to a temporary PNG
        buf = io.BytesIO()
        shap_fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                         facecolor="#0f172a")
        buf.seek(0)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(buf.read())
            tmp_path = tmp.name

        available_w = pdf.w - pdf.l_margin - pdf.r_margin
        pdf.image(tmp_path, x=pdf.l_margin, w=available_w)

        import os
        os.unlink(tmp_path)

    pdf_bytes = bytes(pdf.output())
    return pdf_bytes
