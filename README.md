# AutoML Forge

AutoML Forge is an educational AutoML workbench that helps users inspect a CSV dataset, preprocess mixed data, compare machine-learning models, review predictions and export results. It supports both regression and classification tasks.

## Features
- CSV upload, preview, row/column summary and missing-value checks
- Automatic numeric and categorical preprocessing
- Classification and regression workflows
- Dummy baseline plus multiple candidate estimators
- Classification metrics: Accuracy, weighted Precision, weighted Recall and weighted F1
- Regression metrics: MAE, RMSE and R²
- Model-comparison chart and best-model prediction table
- Downloadable predictions and metrics CSV files
- Two included demo datasets
- Reusable evaluation engine and pytest test suite
- GitHub Actions workflow configured for automated tests
- Project report and output guide under docs/

## Run locally
1. Install Python 3.10 or newer.
2. Clone or download this repository.
3. Optional but recommended: create and activate a virtual environment.
4. Install dependencies: `pip install -r requirements.txt`
5. Start the app: `streamlit run app.py`
6. Open the local URL printed by Streamlit.

## Demo A — Regression
Upload `data/student_performance.csv`, choose `final_score` as the target, select **Regression**, and click **Train and compare models**. Review MAE, RMSE, R², the comparison chart, predictions and CSV export buttons.

## Demo B — Classification
Upload `data/student_risk_classification.csv`, choose `needs_support` as the target, select **Classification**, and click **Train and compare models**. Review accuracy, weighted precision, weighted recall and weighted F1.

The datasets are synthetic teaching examples. They are not suitable for making decisions about real students or other people.

## Run automated tests
Run `pytest -q` from the repository root. The GitHub Actions workflow is configured to run the tests on pushes and pull requests to `main`. Check the Actions tab for the actual result; a workflow file existing does not itself mean a run passed.

## Repository structure
```text
AutoML-Forge/
├── app.py
├── automl_engine.py
├── requirements.txt
├── README.md
├── data/
│   ├── student_performance.csv
│   └── student_risk_classification.csv
├── tests/
│   ├── test_dataset.py
│   └── test_automl_engine.py
├── docs/
│   ├── OUTPUT_GUIDE.md
│   └── PROJECT_REPORT.md
└── .github/workflows/test.yml
```

## Output details
After training, the interface shows the best successful model, a metric table, a bar chart, test-set predictions and CSV downloads. The detailed walkthrough and metric definitions are in [Output Guide](docs/OUTPUT_GUIDE.md).

## Project documentation
- [Mini Project Report](docs/PROJECT_REPORT.md) — problem statement, objectives, architecture, modules, testing, limitations and future scope.
- [ER Diagram / Conceptual Data Model](docs/ER_DIAGRAM.md) — Mermaid ER diagram with entity and relationship explanations. Note: this is a conceptual model; the current app does not persist these entities in database tables.
- [Output Guide](docs/OUTPUT_GUIDE.md) — how to run both demos and interpret the metrics.

## Limitations and responsible use
This is an educational prototype. Scores depend on dataset quality and one train/test split, and may not generalize. Cross-validation, model persistence, stronger leakage checks and domain validation are future improvements. Do not upload sensitive personal or confidential data.
