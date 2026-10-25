# AutoML Forge — Sample Test Output

## Test environment
- Task: Regression
- Dataset: `data/student_performance.csv` (synthetic teaching dataset; 20 rows)
- Target: `final_score`
- Test split: 20%
- Split random state: 42
- Models attempted: 7
- Local evaluation: run using the repository's scikit-learn preprocessing/model configuration.

## Model comparison

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Dummy baseline | 15.0000 | 16.1405 | -0.0328 |
| Linear Regression | 1.9982 | 3.0428 | 0.9633 |
| Ridge Regression | 1.7467 | 2.8859 | 0.9670 |
| Decision Tree | 10.5000 | 12.4700 | 0.3835 |
| Random Forest | 1.7800 | 2.6720 | 0.9717 |
| Extra Trees | 2.8667 | 4.7992 | 0.9087 |
| Hist Gradient Boosting | 15.0000 | 16.1405 | -0.0328 |

**Best model for this split: Random Forest**, selected by lowest RMSE (2.6720). The scores are from a small synthetic dataset and a single test split; they are demonstration results, not evidence of real-world student-performance prediction quality.

## Actual vs predicted — best model

| Actual | Predicted |
|---:|---:|
| 48 | 47.6400 |
| 81 | 80.9667 |
| 88 | 86.3400 |
| 61 | 66.0667 |

The prediction order follows the deterministic test split with random state 42.

## Automated tests
GitHub Actions run **12** for commit `7e25ca31fc89519600f5fa6cf19c7bb4814047b1` completed successfully. The log reports **6 passed in 2.38s**. [View workflow run](https://github.com/SwapnilPatil-1993/AutoML-Forge/actions/runs/40198954594).

## Important distinction
This Markdown file records actual computed sample metrics and predictions; it is not a screenshot of the Streamlit interface. To capture UI screenshots, run `streamlit run app.py`, upload the sample CSV, select `final_score` and Regression, and click **Train and compare models**.
