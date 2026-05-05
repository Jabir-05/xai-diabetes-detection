# Diabetes Risk XAI

Explainable machine-learning project for diabetes risk screening using a soft-voting ensemble, SHAP explanations, batch CSV prediction, PDF reporting, and fairness auditing.

## Features

- Interactive Streamlit dashboard with live patient risk simulation
- Single-patient prediction with SHAP waterfall and force plots
- What-if counterfactual sliders and session risk history
- Batch CSV upload with risk distribution charts and downloadable results
- Global SHAP insights, feature correlations, and dataset exploration
- Fairlearn-based age-group fairness audit
- Downloadable PDF patient report

## How to Run

```bash
cd "/Users/jabirimteyaz/Desktop/Major Project/diabetes_xai"
python3 -m pip install -r requirements.txt
python3 model.py
python3 -m streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal, usually:

```text
http://localhost:8501
```

## Project Structure

```text
app.py                    Main interactive dashboard
model.py                  Training pipeline and artifact generation
utils.py                  Shared preprocessing, model loading, risk gauge
explainer.py              SHAP explanation helpers
fairness.py               Fairlearn audit helpers
report.py                 PDF report generator
llm_summary.py            Local plain-English patient summary generator
pages/1_Single_Patient.py Individual risk assessment workflow
pages/2_Batch_Upload.py   Multi-patient CSV scoring workflow
pages/3_Global_Insights.py Global SHAP and dataset analysis
pages/4_Fairness_Audit.py Fairness and bias audit
models/                   Saved trained model artifacts
data/diabetes.csv         PIMA diabetes dataset
```

## Notes

This project is for academic and research demonstration only. It is not a medical diagnosis system.
