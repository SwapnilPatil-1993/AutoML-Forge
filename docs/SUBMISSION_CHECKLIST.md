# AutoML Forge — Submission Checklist

**Project:** AutoML Forge — Automated Machine Learning Workbench  
**Student:** Swapnil D. Patil  
**Branch:** Information Technology  
**GitHub repository:** https://github.com/SwapnilPatil-1993/AutoML-Forge

## Documents included in this repository

- [Mini Project Report](PROJECT_REPORT.md) — introduction, problem statement, objectives, tools, architecture, modules, demonstration, testing, limitations, future scope and conclusion.
- [Output Guide](OUTPUT_GUIDE.md) — exact steps for trying the regression and classification workflows and explanations of the metrics.
- [Project README](../README.md) — setup instructions, features, demo datasets and test instructions.

## Project contents

- `app.py` — Streamlit user interface and end-to-end workflow.
- `automl_engine.py` — reusable model evaluation engine.
- `data/student_performance.csv` — regression demonstration dataset.
- `data/student_risk_classification.csv` — synthetic classification demonstration dataset.
- `tests/` — automated tests for data and evaluation logic.
- `.github/workflows/test.yml` — GitHub Actions test workflow.

## Verification status

The latest GitHub Actions run available when this checklist was prepared completed with **success**:
https://github.com/SwapnilPatil-1993/AutoML-Forge/actions/runs/40198396135

This verifies the automated checks for that commit; it does not mean the app has been deployed or that screenshots were captured.

## Run the application

1. Install Python 3.10 or newer.
2. Download or clone the repository.
3. In a terminal opened in the project folder, run:
   `pip install -r requirements.txt`
4. Start the application:
   `streamlit run app.py`
5. Open the local URL printed in the terminal.

## Capture genuine output screenshots

Screenshots should be captured after running the program, not fabricated.

### Screenshot A — Regression
1. Upload `data/student_performance.csv`.
2. Select `final_score` as the target and choose **Regression**.
3. Click **Train and compare models**.
4. Capture the visible model metrics table and chart.
5. Capture the prediction table and CSV download buttons.

### Screenshot B — Classification
1. Upload `data/student_risk_classification.csv`.
2. Select `needs_support` as the target and choose **Classification**.
3. Click **Train and compare models**.
4. Capture the metrics table, chart and prediction table.

Exact metric values depend on the data and current software environment. Do not write invented scores into the report. The classification dataset is synthetic and must not be used to make decisions about real students.

## Final hand-in

Submit the GitHub repository link and the report required by the teacher. If a PDF report or screenshots are specifically required, export the report to PDF and add the genuine screenshots after running the app locally. Online deployment is not required for this project.
