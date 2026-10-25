# AutoML Forge — Verified Sample Output

## Test configuration
- Dataset: `data/student_performance.csv` (20 synthetic/demo rows)
- Task: Regression
- Target column: `final_score`
- Test split: 20% (4 test rows)
- Random state: 42
- Models evaluated by the current `automl_engine.py`: 5

## Model comparison

| Model | R² | MAE | RMSE |
|---|---:|---:|---:|
| Ridge Regression | 0.9670 | 1.7467 | 2.8859 |
| Decision Tree | 0.3835 | 10.5000 | 12.4700 |
| Random Forest | 0.9717 | 1.7800 | 2.6720 |
| Extra Trees | 0.9087 | 2.8667 | 4.7992 |
| Gradient Boosting | 0.9391 | 3.3012 | 3.9185 |

The selected model is **Random Forest**, with R² = 0.9717 and RMSE = 2.6720 on this held-out split. This is a small synthetic dataset and a single split; the scores are only a demonstration, not evidence of real-world predictive performance.

## Actual vs predicted (Random Forest)

| Actual | Predicted |
|---:|---:|
| 48 | 47.6400 |
| 81 | 80.9667 |
| 88 | 86.3400 |
| 61 | 66.0667 |

## Automated test status
The latest verified CI run before this output update completed successfully: **6 tests passed in 2.38 seconds**. [View GitHub Actions](https://github.com/SwapnilPatil-1993/AutoML-Forge/actions).

## Files
- [Model metrics CSV](../outputs/regression_model_metrics.csv)
- [Predictions CSV](../outputs/regression_predictions.csv)
- [Model comparison chart (SVG)](../outputs/model_comparison.svg)
- [Actual vs predicted chart (SVG)](../outputs/prediction_comparison.svg)

**Reproducibility note:** These values were calculated using the repository's current regression model definitions and the exact 20-row sample dataset, with scikit-learn's `train_test_split(test_size=0.2, random_state=42)`. The SVGs are generated result visualizations, not screenshots of the Streamlit interface.
