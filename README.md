# AutoML Forge

AutoML Forge is an interactive web application for quick baseline machine-learning experiments on CSV datasets. Built with Python, Streamlit, pandas, NumPy, and scikit-learn.

## Features
- Upload and preview CSV files
- Inspect missing values and dataset dimensions
- Choose a prediction target and classification or regression task
- Automatic preprocessing for numeric and categorical columns
- Compare multiple models against a dummy baseline
- Classification metrics: Accuracy, weighted Precision, weighted Recall, weighted F1
- Regression metrics: MAE, RMSE, R-squared
- Visual model comparison and downloadable metrics/predictions CSVs
- Included sample dataset and a small dataset-validation test

## Run locally
1. Create and activate a Python virtual environment.
2. Install dependencies: pip install -r requirements.txt
3. Start the app: streamlit run app.py
4. Open the local URL printed by Streamlit.

## Deploy to Streamlit Community Cloud
1. Push this repository to GitHub.
2. Sign in at https://share.streamlit.io/ using GitHub.
3. Choose Create app, select this repository and branch main.
4. Set the main file path to app.py, then deploy.
5. Wait for the build to finish and open the generated app URL.

## Included demo dataset
Upload data/student_performance.csv in the app, choose final_score as the target, and select Regression.

## Project structure
- app.py
- requirements.txt
- README.md
- data/student_performance.csv
- tests/test_dataset.py

## Important limitations
This is an educational AutoML prototype. Scores depend on the dataset and a single train/test split. Before real-world decisions, add cross-validation, stronger data-leakage checks, model persistence, authentication, and domain-specific validation. Do not upload sensitive personal or confidential data.
