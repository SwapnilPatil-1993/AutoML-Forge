"""Reusable training and evaluation logic for AutoML Forge."""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    ExtraTreesClassifier, ExtraTreesRegressor,
    RandomForestClassifier, RandomForestRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Impute missing values and encode mixed numeric/categorical features."""
    numeric = features.select_dtypes(include=np.number).columns.tolist()
    categorical = [column for column in features.columns if column not in numeric]
    transformers = []
    if numeric:
        transformers.append((
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]),
            numeric,
        ))
    if categorical:
        transformers.append((
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            categorical,
        ))
    return ColumnTransformer(transformers, remainder="drop")


def model_catalog(task: str) -> dict:
    """Return candidate estimators for classification or regression."""
    if task == "Classification":
        return {
            "Dummy baseline": DummyClassifier(strategy="most_frequent"),
            "Logistic Regression": LogisticRegression(max_iter=1500),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1),
            "Extra Trees": ExtraTreesClassifier(n_estimators=150, random_state=42, n_jobs=-1),
            "Hist Gradient Boosting": HistGradientBoostingClassifier(random_state=42),
        }
    if task == "Regression":
        return {
            "Dummy baseline": DummyRegressor(strategy="mean"),
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(),
            "Decision Tree": DecisionTreeRegressor(random_state=42),
            "Random Forest": RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1),
            "Extra Trees": ExtraTreesRegressor(n_estimators=150, random_state=42, n_jobs=-1),
            "Hist Gradient Boosting": HistGradientBoostingRegressor(random_state=42),
        }
    raise ValueError("task must be 'Classification' or 'Regression'")


def evaluate_models(
    frame: pd.DataFrame,
    target: str,
    task: str,
    test_fraction: float = 0.2,
) -> dict:
    """Train candidates on a reproducible split and return metrics and predictions."""
    if target not in frame.columns:
        raise ValueError("Target column was not found in the dataset.")
    data = frame.dropna(subset=[target]).copy()
    if len(data) < 10:
        raise ValueError("At least 10 rows with a non-missing target are recommended.")
    X = data.drop(columns=[target])
    y = data[target]
    if X.shape[1] == 0:
        raise ValueError("Keep at least one feature column besides the target.")

    stratify = None
    if task == "Classification":
        if y.nunique() < 2:
            raise ValueError("Classification requires at least two target classes.")
        if y.value_counts().min() >= 2 and y.nunique() < 30:
            stratify = y
    elif task == "Regression":
        y = pd.to_numeric(y, errors="coerce")
        valid = y.notna()
        X, y = X.loc[valid], y.loc[valid]
        if len(y) < 10:
            raise ValueError("Regression target must contain at least 10 numeric values.")
    else:
        raise ValueError("Choose Classification or Regression.")

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_fraction, random_state=42, stratify=stratify
        )
    except ValueError as exc:
        raise ValueError(f"Unable to create train/test split: {exc}") from exc

    rows, fitted = [], {}
    for name, estimator in model_catalog(task).items():
        try:
            pipe = Pipeline([
                ("preprocess", build_preprocessor(X_train)),
                ("model", estimator),
            ])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pipe.fit(X_train, y_train)
                prediction = pipe.predict(X_test)
            if task == "Classification":
                scores = {
                    "Accuracy": float(accuracy_score(y_test, prediction)),
                    "Precision (weighted)": float(precision_score(y_test, prediction, average="weighted", zero_division=0)),
                    "Recall (weighted)": float(recall_score(y_test, prediction, average="weighted", zero_division=0)),
                    "F1 (weighted)": float(f1_score(y_test, prediction, average="weighted", zero_division=0)),
                }
                rank = scores["F1 (weighted)"]
            else:
                rmse = float(np.sqrt(mean_squared_error(y_test, prediction)))
                scores = {
                    "MAE": float(mean_absolute_error(y_test, prediction)),
                    "RMSE": rmse,
                    "R²": float(r2_score(y_test, prediction)),
                }
                rank = -rmse
            rows.append({"Model": name, **scores, "_rank": rank, "Status": "Success"})
            fitted[name] = {"pipeline": pipe, "predictions": prediction}
        except Exception as exc:
            rows.append({"Model": name, "_rank": -np.inf, "Status": f"Skipped: {str(exc)[:140]}"})

    results = pd.DataFrame(rows).sort_values("_rank", ascending=False).reset_index(drop=True)
    successful = results[results["Status"] == "Success"]
    if successful.empty:
        raise RuntimeError("No models trained successfully. Review the dataset and target.")
    best_name = successful.iloc[0]["Model"]
    return {
        "results": results,
        "best_name": best_name,
        "best_pipeline": fitted[best_name]["pipeline"],
        "best_predictions": fitted[best_name]["predictions"],
        "X_test": X_test,
        "y_test": y_test,
        "y_train": y_train,
        "task": task,
    }
