# AutoML Forge — ER Diagram / Data Model

AutoML Forge is an ML application rather than a conventional database-backed CRUD system. It does not currently persist users, datasets, or trained models in a relational database. The following conceptual ER-style model documents the entities handled during one experiment.

```mermaid
erDiagram
    DATASET ||--|{ FEATURE : contains
    DATASET ||--|| TARGET : defines
    EXPERIMENT ||--|| DATASET : uses
    EXPERIMENT ||--|{ MODEL_RUN : evaluates
    MODEL_RUN ||--|| METRIC_SET : produces
    MODEL_RUN ||--o{ PREDICTION : generates

    DATASET {
        string dataset_id PK
        string file_name
        int row_count
        int column_count
        int missing_cell_count
    }
    FEATURE {
        string feature_id PK
        string dataset_id FK
        string feature_name
        string data_type
    }
    TARGET {
        string target_id PK
        string dataset_id FK
        string target_column
        string task_type
    }
    EXPERIMENT {
        string experiment_id PK
        string dataset_id FK
        float test_fraction
        int random_state
    }
    MODEL_RUN {
        string model_run_id PK
        string experiment_id FK
        string model_name
        string status
        boolean selected_best
    }
    METRIC_SET {
        string metric_set_id PK
        string model_run_id FK
        float accuracy
        float f1_weighted
        float mae
        float rmse
        float r_squared
    }
    PREDICTION {
        string prediction_id PK
        string model_run_id FK
        string actual_value
        string predicted_value
        boolean matched
    }
```

## Explanation
- **Dataset** describes the uploaded CSV and its basic quality summary.
- **Feature** represents a predictor column; a dataset contains one or more features.
- **Target** identifies the column to predict and whether the experiment is classification or regression.
- **Experiment** represents one configured train/test evaluation of a dataset.
- **Model Run** represents one candidate estimator evaluated during the experiment.
- **Metric Set** stores the evaluation measures conceptually produced by a model run. Classification uses accuracy, weighted precision/recall/F1; regression uses MAE, RMSE and R².
- **Prediction** represents an actual/predicted pair from the held-out test set.

**Implementation note:** This diagram is a conceptual ER-style model for the report. The current application processes data in memory and offers CSV downloads; it does not create these entities as database tables or persist them between sessions.
