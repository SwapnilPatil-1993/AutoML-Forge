# AutoML Forge — Output Guide

This guide describes the output panels produced after a CSV is uploaded and Train and compare models is clicked. Exact scores depend on the dataset and test split; no fixed score is guaranteed.

## Workflow A: Regression — student performance
1. Upload data/student_performance.csv.
2. Set Target column to final_score.
3. Select Regression and keep the default 20% test data.
4. Click Train and compare models.

The application displays:
- Best model on this test split — candidate with the lowest RMSE among successful models.
- Model comparison — Dummy baseline, Linear Regression, Ridge Regression, Decision Tree, Random Forest, Extra Trees and Hist Gradient Boosting. Metrics include MAE, RMSE and R².
- Model comparison chart.
- Best model predictions with Actual, Predicted and Match columns.
- Download controls for prediction rows and model metrics as CSV files.

## Workflow B: Classification — student support indicator
1. Upload data/student_risk_classification.csv.
2. Set Target column to needs_support.
3. Select Classification and keep the default 20% test data.
4. Click Train and compare models.

The application displays Accuracy, weighted Precision, weighted Recall and weighted F1. The top model is selected using weighted F1. This synthetic dataset is for software demonstration only and must not be used to make decisions about real students.

## Understanding the metrics
- Accuracy: fraction of predictions that are correct.
- Precision: among predicted members of a class, how many were correct; weighted across classes.
- Recall: among actual members of a class, how many were found; weighted across classes.
- F1: combines precision and recall; weighted across classes.
- MAE: average absolute prediction error; lower is better.
- RMSE: square root of average squared error; lower is better and larger errors count more.
- R²: compares model performance with predicting the target mean; values near 1 indicate a better fit; negative values can occur.

## Troubleshooting
- Ensure the file is CSV with a header row and at least two columns.
- Use a target with at least 10 usable rows.
- Classification requires at least two classes.
- Regression requires a numeric target.
- If a model is skipped, read its Status message in the comparison table.
