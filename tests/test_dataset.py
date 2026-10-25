import pandas as pd
from pathlib import Path

def test_demo_dataset_has_target_and_features():
    path = Path(__file__).parents[1] / "data" / "student_performance.csv"
    df = pd.read_csv(path)
    assert len(df) >= 10
    assert "final_score" in df.columns
    assert df.shape[1] >= 2
    assert df["final_score"].notna().all()
