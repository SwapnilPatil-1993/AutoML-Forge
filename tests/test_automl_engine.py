import pandas as pd
import pytest

from automl_engine import build_preprocessor, evaluate_models, model_catalog


def sample_regression_data():
    return pd.DataFrame({
        "hours": [1, 2, 3, 4, 5, 6, 7, 8, 2, 4, 6, 8, 3, 5, 7, 1, 4, 6, 8, 5],
        "attendance": [55, 60, 65, 70, 75, 80, 85, 90, 62, 72, 82, 92, 68, 78, 88, 58, 73, 84, 95, 79],
        "course": ["IT", "CS", "AI", "IT", "CS"] * 4,
        "score": [40, 45, 52, 58, 65, 72, 78, 90, 47, 60, 76, 94, 55, 68, 86, 42, 62, 80, 96, 70],
    })


def test_catalog_contains_baseline_and_candidates():
    assert "Dummy baseline" in model_catalog("Classification")
    assert "Random Forest" in model_catalog("Regression")


def test_preprocessor_supports_numeric_and_categorical_data():
    frame = sample_regression_data().drop(columns=["score"])
    transformed = build_preprocessor(frame).fit_transform(frame)
    assert transformed.shape[0] == len(frame)
    assert transformed.shape[1] >= 3


def test_regression_evaluation_returns_metrics_and_predictions():
    result = evaluate_models(sample_regression_data(), "score", "Regression", 0.2)
    assert result["best_name"]
    assert len(result["best_predictions"]) == len(result["y_test"])
    assert "RMSE" in result["results"].columns
    assert (result["results"]["Status"] == "Success").any()


def test_invalid_target_is_reported():
    with pytest.raises(ValueError, match="Target column"):
        evaluate_models(sample_regression_data(), "missing", "Regression")


def test_classification_evaluation_returns_f1_metric():
    frame = pd.DataFrame({
        "hours": [1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,2,4,6,8],
        "course": ["IT","CS","AI","IT","CS"] * 4,
        "support": ["Yes","Yes","Yes","No","No","No","No","No","Yes","Yes","Yes","No","No","No","No","No","Yes","No","No","No"],
    })
    result = evaluate_models(frame, "support", "Classification", 0.2)
    assert "F1 (weighted)" in result["results"].columns
