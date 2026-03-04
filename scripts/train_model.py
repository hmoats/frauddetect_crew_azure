#!/usr/bin/env python
"""Offline model training script.

Run once (or periodically) to produce the pre-trained artifacts that the
inference flow depends on.  No LLM or CrewAI dependency -- pure Python / sklearn.

Usage:
    python scripts/train_model.py
    # or, if registered in pyproject.toml:
    uv run train
"""

import io
import json
import os
import sys
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

GITHUB_URL = (
    "https://raw.githubusercontent.com/etl919191/kaggle/main/synthetic_customer_data.csv"
)
TARGET_COLUMN = "Is_Fraud_AML"
EXCLUDE_PREFIXES = ["Risk_Type"]

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(ROOT_DIR, "src", "frauddetect_crew", "models")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")


def main() -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Downloading dataset ...")
    response = requests.get(GITHUB_URL, timeout=120)
    response.raise_for_status()
    df = pd.read_csv(io.StringIO(response.text))
    print(f"  {len(df)} rows, {len(df.columns)} columns")

    if TARGET_COLUMN not in df.columns:
        sys.exit(f"Target column '{TARGET_COLUMN}' not found. Available: {list(df.columns)}")

    drop_cols = [TARGET_COLUMN]
    for col in df.columns:
        if any(col.startswith(p) for p in EXCLUDE_PREFIXES):
            drop_cols.append(col)

    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = pd.get_dummies(df[feature_cols])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

    print("Training RandomForest with GridSearchCV (full grid, 5-fold CV) ...")
    model = RandomForestClassifier(random_state=42)
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10],
    }
    grid_search = GridSearchCV(
        model, param_grid, cv=5, scoring="f1_weighted", n_jobs=-1, verbose=0,
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    print(f"  Best params: {grid_search.best_params_}")

    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    print(f"  Accuracy: {acc:.4f}  F1: {f1:.4f}  Precision: {prec:.4f}  Recall: {rec:.4f}")

    # --- Save model artifacts to models/ ---
    model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    joblib.dump(best_model, model_path)
    print(f"  Model saved to {model_path}")

    feature_cols_path = os.path.join(MODELS_DIR, "feature_columns.json")
    with open(feature_cols_path, "w") as f:
        json.dump(list(X.columns), f)
    print(f"  Feature columns saved ({len(X.columns)} features)")

    feat_imp = pd.Series(
        best_model.feature_importances_, index=X.columns,
    ).sort_values(ascending=False)
    top_features = feat_imp.head(10).round(4).to_dict()

    metrics = {
        "accuracy": round(float(acc), 4),
        "f1_score": round(float(f1), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "model_type": "RandomForest",
        "best_params": grid_search.best_params_,
        "total_samples": len(df),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "top_features": top_features,
    }
    metrics_path = os.path.join(MODELS_DIR, "model_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved to {metrics_path}")

    # --- Save sample transactions for quick inference demos ---
    fraud_sample = df[df[TARGET_COLUMN] == 1].sample(n=min(10, len(df[df[TARGET_COLUMN] == 1])), random_state=42)
    clean_sample = df[df[TARGET_COLUMN] == 0].sample(n=10, random_state=42)
    sample_df = pd.concat([fraud_sample, clean_sample]).drop(
        columns=[TARGET_COLUMN, "Risk_Type"], errors="ignore",
    )
    sample_records = sample_df.to_dict(orient="records")
    sample_path = os.path.join(MODELS_DIR, "sample_transactions.json")
    with open(sample_path, "w") as f:
        json.dump(sample_records, f, indent=2)
    print(f"  Sample transactions saved ({len(sample_records)} records)")

    # --- Optional diagnostic outputs to output/ ---
    report_text = classification_report(y_test, y_pred, output_dict=False)
    with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
        f.write(report_text)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    top_n = feat_imp.head(15)
    sns.barplot(x=top_n.values, y=top_n.index)
    plt.title("Top 15 Feature Importances")
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"))
    plt.close()

    print("\nTraining complete. Artifacts in models/ are ready to commit.")


if __name__ == "__main__":
    main()
