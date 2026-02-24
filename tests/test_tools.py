"""Tests for the transaction lookup and model scoring tools."""

import json
import os
import tempfile

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from frauddetect_crew.tools.transaction_lookup_tool import TransactionLookupTool
from frauddetect_crew.tools.model_scoring_tool import ModelScoringTool


class TestTransactionLookupTool:
    def test_lookup_existing_customer(self):
        tool = TransactionLookupTool()
        result = json.loads(
            tool._run(customer_id="d56303bd-a099-4650-bfcb-61d047b74e13")
        )
        assert result["status"] == "found"
        assert result["record"]["Customer_ID"] == "d56303bd-a099-4650-bfcb-61d047b74e13"

    def test_lookup_missing_customer(self):
        tool = TransactionLookupTool()
        result = json.loads(tool._run(customer_id="nonexistent-id"))
        assert result["status"] == "not_found"
        assert "available_ids" in result
        assert len(result["available_ids"]) > 0

    def test_lookup_returns_numeric_fields(self):
        tool = TransactionLookupTool()
        result = json.loads(
            tool._run(customer_id="d56303bd-a099-4650-bfcb-61d047b74e13")
        )
        record = result["record"]
        assert isinstance(record["Transaction_Amount"], float)
        assert isinstance(record["Age"], float)


class TestModelScoringTool:
    def test_scoring_with_invalid_json(self):
        tool = ModelScoringTool()
        result = json.loads(tool._run(transaction_json="not valid json"))
        assert result["status"] == "error"
        assert "Invalid JSON" in result["message"]

    def test_scoring_with_sample_record(self):
        """Score a record from the sample data against the real model."""
        models_dir = os.path.join(
            os.path.dirname(__file__), "..", "models"
        )
        model_path = os.path.join(models_dir, "best_model.pkl")
        if not os.path.exists(model_path):
            return

        tool = ModelScoringTool()
        record = {
            "Customer_ID": "d56303bd-a099-4650-bfcb-61d047b74e13",
            "Transaction_Amount": 6.36,
            "Account_Balance": 24444.05,
            "Merchant_Category": "Digital Services",
            "Device_Type": "Mobile",
        }
        result = json.loads(tool._run(transaction_json=json.dumps(record)))
        assert result["status"] == "scored"
        assert result["verdict"] in ("FRAUD", "APPROVED")
        assert 0.0 <= result["risk_score"] <= 1.0
