# AutoML Forge — Mini Project Report

## 1. Introduction
Machine-learning projects often require repeated steps: preparing data, selecting an algorithm, training models and comparing results. AutoML Forge is an educational Python application that combines these steps in a guided interface. It allows a user to upload a CSV dataset, inspect data quality, choose a target, select classification or regression, and compare candidate models.

## 2. Problem Statement
Beginners may find it difficult to compare several machine-learning algorithms consistently. Manual preprocessing can cause errors when datasets contain missing values or categorical columns. The project aims to provide a compact workbench that automates common preparation and evaluation tasks while keeping the workflow visible.

## 3. Objectives
- Build a user-friendly CSV-based machine-learning interface.
- Summarize dataset dimensions and missing values before training.
- Support classification and regression.
- Impute missing numeric and categorical values automatically.
- Scale numeric features and encode categorical features.
- Compare candidate models against a dummy baseline.
- Display metrics and charts.
- Export predictions and metric tables to CSV.
- Include sample datasets, automated tests and documentation.

## 4. Tools and Technologies
- Python: application and machine-learning logic.
- Streamlit: interactive web interface.
- pandas and NumPy: tabular data handling and numerical operations.
- scikit-learn: preprocessing, models, train/test split and evaluation.
- pytest: automated tests.
- GitHub Actions: continuous integration workflow configured to run tests on pushes and pull requests.

## 5. System Architecture
The user uploads a CSV through Streamlit. The application validates the dataset, displays a preview and missing-value summary, and asks the user to select a target and task. A preprocessing pipeline imputes missing values, scales numeric columns and one-hot encodes categorical columns. Multiple estimators are trained using a reproducible train/test split. The application compares metrics, selects the best successful model according to weighted F1 for classification or RMSE for regression, then presents predictions and CSV export options.

Architecture flow: User → CSV upload → Dataset validation and preview → Target/task selection → Preprocessing pipeline → Train/test split → Candidate model training → Metrics and chart → Prediction and metrics CSV export.

## 6. ER Diagram / Conceptual Data Model
The conceptual ER-style model is documented in [ER_DIAGRAM.md](ER_DIAGRAM.md). It describes Dataset, Feature, Target, Experiment, Model Run, Metric Set and Prediction entities and their relationships. This diagram documents the logical information handled during an experiment; it is not a claim that the application currently stores these as relational database tables. The current version processes data in memory and provides CSV downloads.

## 7. Main Modules
1. Data input and validation: reads CSV data and checks that it has sufficient rows and columns.
2. Data quality summary: reports row count, column count and missing cells.
3. Preprocessing: median imputation and scaling for numeric data; most-frequent imputation and one-hot encoding for categorical data.
4. Model catalogue: classification and regression estimators, including a dummy baseline.
5. Evaluation: classification uses accuracy, weighted precision, weighted recall and weighted F1. Regression uses MAE, RMSE and R².
6. Results and export: sorts results, charts metrics, shows actual and predicted values, and provides downloadable CSV files.
7. Quality checks: tests cover dataset structure, preprocessing, model catalogue, evaluation and invalid targets.

## 7. Sample Demonstration
### Regression
Upload data/student_performance.csv, choose final_score as target and choose Regression. Output includes a model-comparison table with MAE, RMSE and R², a chart, the selected model name, prediction rows and two CSV download controls.

### Classification
Upload data/student_risk_classification.csv, choose needs_support as target and choose Classification. Output includes accuracy, weighted precision, weighted recall and weighted F1 for each successful model, plus a chart and prediction download.

Exact scores are data-dependent. Record actual results from a local run if numerical results are required; do not treat illustrative values as measured results.

## 8. Testing and Quality Assurance
The repository includes tests for dataset structure, mixed-type preprocessing, regression evaluation, classification evaluation and invalid target handling. A GitHub Actions workflow is configured to install dependencies and run pytest on pushes and pull requests to main. The workflow configuration is present; confirm a successful run from the GitHub Actions tab before claiming tests passed.

## 9. Limitations
- The current version evaluates one train/test split rather than cross-validation.
- Results depend on dataset size, quality, target selection and class balance.
- Some estimators may be skipped for unsupported or unsuitable data; the app reports the status.
- Trained models are not saved between sessions.
- This is an educational prototype, not a validated decision-making system.
- Sensitive or confidential data should not be uploaded.

## 10. Future Scope
- Add cross-validation and configurable scoring strategies.
- Add feature-importance and error-analysis views.
- Add model persistence and reproducible experiment history.
- Add richer data validation and leakage warnings.
- Add PDF/HTML experiment report export.
- Add responsible-AI checks and dataset documentation.

## 11. Conclusion
AutoML Forge demonstrates a beginner-friendly machine-learning workflow: data inspection, preprocessing, training, comparison, evaluation and export. Its modular structure, sample datasets, tests and documentation make it suitable for a college mini-project demonstration. Before presenting numerical performance claims, run the app locally and capture actual results.
