"""End-to-end supervised classification project.

Run from any working directory with: python src/train_model.py
The script reads the dataset shipped in ../data and writes reproducible
metrics and model artifacts to ../outputs.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    make_scorer,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "breast_cancer_wisconsin_diagnostic.csv"
OUTPUT_DIR = ROOT / "outputs"
RANDOM_STATE = 42
TEST_SIZE = 0.20
POSITIVE_CLASS = "malignant"
TARGET_COLUMN = "diagnosis"


def build_models() -> dict[str, object]:
    """Return baseline and candidate classifiers with leakage-safe preprocessing."""
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=5000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "RBF Support Vector Machine": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    SVC(kernel="rbf", class_weight="balanced", random_state=RANDOM_STATE),
                ),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Majority-Class Baseline": DummyClassifier(strategy="most_frequent"),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Keep the project data/ folder intact."
        )

    # 1. Read and validate the dataset before modeling.
    data = pd.read_csv(DATA_PATH)
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Expected target column {TARGET_COLUMN!r}.")
    if data[TARGET_COLUMN].isna().any():
        raise ValueError("The target contains missing values; investigate before training.")
    unexpected_labels = set(data[TARGET_COLUMN].unique()) - {"benign", "malignant"}
    if unexpected_labels:
        raise ValueError(f"Unexpected target labels: {unexpected_labels}")

    X = data.drop(columns=[TARGET_COLUMN])
    y = (data[TARGET_COLUMN] == POSITIVE_CLASS).astype("int64")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in X.dtypes):
        raise TypeError("All model input features must be numeric.")
    missing_features = X.isna().sum()
    if missing_features.any():
        raise ValueError(f"Missing feature values found: {missing_features[missing_features > 0].to_dict()}")
    if X.shape[1] != 30:
        raise ValueError(f"Expected 30 diagnostic features; found {X.shape[1]}.")
    if y.nunique() != 2:
        raise ValueError("This project expects exactly two target classes.")

    # 2. Hold out a stratified test set. It is not used for model selection.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # 3. Compare candidate models by five-fold CV on training data only.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "roc_auc": "roc_auc",
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
    }
    models = build_models()
    cv_rows: list[dict[str, float | str]] = []
    for model_name, estimator in models.items():
        scores = cross_validate(
            estimator,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
            error_score="raise",
            n_jobs=-1,
        )
        row: dict[str, float | str] = {"model": model_name}
        for metric_name in scoring:
            values = scores[f"test_{metric_name}"]
            row[f"cv_{metric_name}_mean"] = float(values.mean())
            row[f"cv_{metric_name}_std"] = float(values.std(ddof=1))
        cv_rows.append(row)

    comparison = pd.DataFrame(cv_rows).sort_values(
        by="cv_roc_auc_mean", ascending=False, ignore_index=True
    )
    comparison.to_csv(OUTPUT_DIR / "cross_validation_comparison.csv", index=False)

    # Choose the highest mean ROC-AUC among non-baseline models.
    candidate_rows = comparison[comparison["model"] != "Majority-Class Baseline"]
    selected_name = str(candidate_rows.iloc[0]["model"])
    final_model = models[selected_name]
    final_model.fit(X_train, y_train)

    # 4. Evaluate the selected model once on the untouched holdout set.
    y_pred = final_model.predict(X_test)
    if hasattr(final_model, "predict_proba"):
        positive_score = final_model.predict_proba(X_test)[:, 1]
    else:
        positive_score = final_model.decision_function(X_test)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    metrics = {
        "selected_model": selected_name,
        "positive_class": POSITIVE_CLASS,
        "decision_rule": "Estimator default classification threshold (0.5 for logistic probability)",
        "test_size": int(len(y_test)),
        "test_metrics": {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision_malignant": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall_malignant": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1_malignant": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, positive_score)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
            "specificity_benign": float(cm[0, 0] / cm[0].sum()),
        },
        "confusion_matrix": {
            "labels_order": ["benign (0)", "malignant (1)"],
            "rows_actual_columns_predicted": cm.tolist(),
        },
        "classification_report": classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["benign", "malignant"],
            output_dict=True,
            zero_division=0,
        ),
    }
    with (OUTPUT_DIR / "test_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    report_text = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["benign", "malignant"],
        digits=4,
        zero_division=0,
    )
    (OUTPUT_DIR / "classification_report.txt").write_text(report_text, encoding="utf-8")

    cm_frame = pd.DataFrame(
        cm,
        index=["actual_benign", "actual_malignant"],
        columns=["predicted_benign", "predicted_malignant"],
    )
    cm_frame.to_csv(OUTPUT_DIR / "confusion_matrix.csv", index=True)

    predictions = pd.DataFrame(
        {
            "actual_diagnosis": ["malignant" if value else "benign" for value in y_test],
            "predicted_diagnosis": ["malignant" if value else "benign" for value in y_pred],
            "malignancy_score": positive_score,
        },
        index=X_test.index,
    ).sort_index()
    predictions.index.name = "source_row_index"
    predictions.to_csv(OUTPUT_DIR / "test_predictions.csv")

    # Save the fitted preprocessing + estimator as one portable pipeline.
    joblib.dump(final_model, OUTPUT_DIR / "selected_model.joblib")

    metadata = {
        "dataset_file": str(DATA_PATH.relative_to(ROOT)),
        "dataset_rows": int(len(data)),
        "feature_count": int(X.shape[1]),
        "class_counts": {key: int(value) for key, value in data[TARGET_COLUMN].value_counts().items()},
        "missing_values_total": int(data.isna().sum().sum()),
        "train_rows": int(len(y_train)),
        "test_rows": int(len(y_test)),
        "test_size_fraction": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "cross_validation": "5-fold StratifiedKFold; shuffled; training set only",
        "model_selection_metric": "mean cross-validated ROC-AUC",
        "scikit_learn_version": sklearn.__version__,
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_source": "UCI Machine Learning Repository, Breast Cancer Wisconsin (Diagnostic)",
    }
    with (OUTPUT_DIR / "run_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print("Run complete.")
    print(f"Rows: {len(data)} | Features: {X.shape[1]} | Train/test: {len(y_train)}/{len(y_test)}")
    print(f"Selected by mean 5-fold CV ROC-AUC: {selected_name}")
    print("Cross-validation comparison: outputs/cross_validation_comparison.csv")
    print("Held-out test metrics: outputs/test_metrics.json")
    print("Confusion matrix (actual rows; predicted columns; benign then malignant):")
    print(cm_frame.to_string())
    print("Held-out test classification report:")
    print(report_text)
    print(f"All outputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
