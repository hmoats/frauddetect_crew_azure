import json
import os
from typing import Type

import joblib
import pandas as pd
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from frauddetect_crew.paths import MODELS_DIR


class ModelScoringInput(BaseModel):
    """Input schema for the model scoring tool."""

    transaction_json: str = Field(
        ...,
        description=(
            "JSON string of the transaction record to score. "
            "Must include the same fields as the training data "
            "(e.g. Customer_ID, Transaction_Amount, Account_Balance, etc.)."
        ),
    )


class ModelScoringTool(BaseTool):
    """Scores a transaction record against the pre-trained fraud detection model."""

    name: str = "fraud_model_scorer"
    description: str = (
        "Scores a single transaction record against the pre-trained RandomForest "
        "fraud detection model. Accepts the transaction as a JSON string and returns "
        "the predicted label (0 = legitimate, 1 = fraud) and the fraud probability "
        "score (0.0 to 1.0). Also returns the model's accuracy and F1 metrics."
    )
    args_schema: Type[BaseModel] = ModelScoringInput

    def _run(self, transaction_json: str) -> str:
        model_path = os.path.join(MODELS_DIR, "best_model.pkl")
        feature_cols_path = os.path.join(MODELS_DIR, "feature_columns.json")
        metrics_path = os.path.join(MODELS_DIR, "model_metrics.json")

        for path, label in [
            (model_path, "Trained model"),
            (feature_cols_path, "Feature columns"),
            (metrics_path, "Model metrics"),
        ]:
            if not os.path.exists(path):
                return json.dumps({
                    "status": "error",
                    "message": f"{label} not found at {path}. Run scripts/train_model.py first.",
                })

        try:
            record = json.loads(transaction_json)
        except json.JSONDecodeError as e:
            return json.dumps({"status": "error", "message": f"Invalid JSON: {e}"})

        model = joblib.load(model_path)
        with open(feature_cols_path) as f:
            feature_columns = json.load(f)
        with open(metrics_path) as f:
            model_metrics = json.load(f)

        record_df = pd.DataFrame([record])
        record_df = pd.get_dummies(record_df)
        record_df = record_df.reindex(columns=feature_columns, fill_value=0)

        prediction = int(model.predict(record_df)[0])
        proba = model.predict_proba(record_df)
        risk_score = float(proba[0][1]) if proba.shape[1] == 2 else float(proba[0].max())

        return json.dumps({
            "status": "scored",
            "predicted_label": prediction,
            "risk_score": round(risk_score, 4),
            "verdict": "FRAUD" if prediction == 1 else "APPROVED",
            "model_type": model_metrics.get("model_type", "RandomForest"),
            "model_accuracy": model_metrics.get("accuracy"),
            "model_f1_score": model_metrics.get("f1_score"),
            "top_features": model_metrics.get("top_features", {}),
        }, indent=2)
