import io
import warnings
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

st.set_page_config(page_title="AutoML Forge", page_icon="⚙️", layout="wide")
st.title("⚙️ AutoML Forge")
st.caption("An interactive AutoML workbench — upload a CSV, compare models, and export predictions.")
st.markdown("Choose a target column and task type. AutoML Forge preprocesses mixed data, trains baseline and candidate models, and compares their scores.")

@st.cache_data
def load_csv(raw):
    return pd.read_csv(io.BytesIO(raw))

def make_preprocessor(X):
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical))
    return ColumnTransformer(transformers, remainder="drop")

def get_models(task):
    if task == "Classification":
        return {
            "Dummy baseline": DummyClassifier(strategy="most_frequent"),
            "Logistic Regression": LogisticRegression(max_iter=1500),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1),
            "Extra Trees": ExtraTreesClassifier(n_estimators=150, random_state=42, n_jobs=-1),
            "Hist Gradient Boosting": HistGradientBoostingClassifier(random_state=42),
        }
    return {
        "Dummy baseline": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1),
        "Extra Trees": ExtraTreesRegressor(n_estimators=150, random_state=42, n_jobs=-1),
        "Hist Gradient Boosting": HistGradientBoostingRegressor(random_state=42),
    }

uploaded = st.file_uploader("Upload a CSV dataset", type=["csv"])
if uploaded is None:
    st.info("No file uploaded yet. Try the included demo dataset at data/student_performance.csv.")
    st.stop()

try:
    df = load_csv(uploaded.getvalue())
except Exception as exc:
    st.error(f"Could not read this CSV: {exc}")
    st.stop()

if df.empty or len(df.columns) < 2:
    st.error("Please upload a dataset with at least two columns and some rows.")
    st.stop()

with st.expander("Dataset preview and quality checks", expanded=True):
    a, b, c = st.columns(3)
    a.metric("Rows", f"{len(df):,}")
    b.metric("Columns", len(df.columns))
    c.metric("Missing cells", int(df.isna().sum().sum()))
    st.dataframe(df.head(100), use_container_width=True)
    missing = df.isna().sum().rename("Missing values").to_frame()
    st.dataframe(missing, use_container_width=True)

left, right = st.columns([1, 2])
with left:
    target = st.selectbox("Target column to predict", options=df.columns.tolist(), index=len(df.columns)-1)
    task = st.radio("Machine learning task", ["Classification", "Regression"], horizontal=False)
    test_size = st.slider("Test data (%)", 10, 40, 20, 5)
    run_button = st.button("🚀 Train and compare models", type="primary", use_container_width=True)
with right:
    st.write("**What happens when you train?**")
    st.write("1. Separates features and target. 2. Fills missing numeric/categorical values. 3. Scales numeric features and one-hot encodes categories. 4. Evaluates several models on a held-out test split.")
    st.warning("Use datasets you are authorized to process. Results depend on data quality and may not generalize to new data.")

if run_button:
    data = df.dropna(subset=[target]).copy()
    if len(data) < 10:
        st.error("At least 10 rows with a non-missing target are recommended.")
        st.stop()
    X = data.drop(columns=[target])
    y = data[target]
    if X.shape[1] == 0:
        st.error("Select a target while keeping at least one feature column.")
        st.stop()
    if task == "Classification":
        if y.nunique() < 2:
            st.error("Classification requires at least two target classes.")
            st.stop()
        stratify = y if y.value_counts().min() >= 2 and y.nunique() < 30 else None
    else:
        y = pd.to_numeric(y, errors="coerce")
        valid = y.notna()
        X, y = X.loc[valid], y.loc[valid]
        if len(y) < 10:
            st.error("Regression target must contain at least 10 numeric values.")
            st.stop()
        stratify = None
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42, stratify=stratify)
    except ValueError as exc:
        st.error(f"Could not create the train/test split: {exc}. Try a larger dataset or a different test percentage.")
        st.stop()

    rows, fitted = [], {}
    progress = st.progress(0, text="Preparing models...")
    models = get_models(task)
    for i, (name, estimator) in enumerate(models.items(), start=1):
        try:
            pipe = Pipeline([("preprocess", make_preprocessor(X_train)), ("model", estimator)])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pipe.fit(X_train, y_train)
                pred = pipe.predict(X_test)
            if task == "Classification":
                scores = {
                    "Accuracy": accuracy_score(y_test, pred),
                    "Precision (weighted)": precision_score(y_test, pred, average="weighted", zero_division=0),
                    "Recall (weighted)": recall_score(y_test, pred, average="weighted", zero_division=0),
                    "F1 (weighted)": f1_score(y_test, pred, average="weighted", zero_division=0),
                }
                primary = scores["F1 (weighted)"]
            else:
                rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
                scores = {"MAE": mean_absolute_error(y_test, pred), "RMSE": rmse, "R²": r2_score(y_test, pred)}
                primary = -rmse
            rows.append({"Model": name, **scores, "_rank": primary, "Status": "Success"})
            fitted[name] = (pipe, pred)
        except Exception as exc:
            rows.append({"Model": name, "_rank": -np.inf, "Status": f"Skipped: {str(exc)[:140]}"})
        progress.progress(i/len(models), text=f"Evaluated {i}/{len(models)} models")
    progress.empty()
    results = pd.DataFrame(rows).sort_values("_rank", ascending=False)
    successful = results[results["Status"] == "Success"]
    if successful.empty:
        st.error("No models trained successfully. Check the target column and dataset.")
        st.dataframe(results.drop(columns=["_rank"]), use_container_width=True)
        st.stop()
    best_name = successful.iloc[0]["Model"]
    best_pipe, best_pred = fitted[best_name]
    st.success(f"Best model on this test split: {best_name}")
    st.subheader("Model comparison")
    st.dataframe(results.drop(columns=["_rank"]), use_container_width=True)
    chart_cols = [col for col in results.columns if col in (["Accuracy", "Precision (weighted)", "Recall (weighted)", "F1 (weighted)"] if task == "Classification" else ["MAE", "RMSE", "R²"])]
    if chart_cols:
        st.bar_chart(results.set_index("Model")[chart_cols])
    predictions = pd.DataFrame({"Actual": np.asarray(y_test), "Predicted": np.asarray(best_pred)})
    predictions["Match"] = predictions["Actual"].astype(str) == predictions["Predicted"].astype(str)
    st.subheader("Best model predictions")
    st.dataframe(predictions.head(100), use_container_width=True)
    st.download_button("⬇️ Download predictions CSV", predictions.to_csv(index=False).encode("utf-8"), "automl_predictions.csv", "text/csv")
    st.download_button("⬇️ Download model metrics CSV", successful.drop(columns=["_rank"]).to_csv(index=False).encode("utf-8"), "automl_model_metrics.csv", "text/csv")
    st.caption("Evaluation is performed on a held-out split, not a guarantee of real-world performance. For serious use, add cross-validation, leakage checks, and domain review.")
else:
    st.subheader("Quick start")
    st.write("Use the sample dataset in the repository: download data/student_performance.csv, then upload it above. Choose final_score as the target and Regression as the task.")
