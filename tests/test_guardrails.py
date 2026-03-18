"""Tests for guardrail functions."""

import json
from unittest.mock import MagicMock

from frauddetect_crew.crew import (
    validate_analysis_has_tool_data,
    validate_required_fields,
    validate_verdict_risk_alignment,
)


def _make_task_output(raw: str) -> MagicMock:
    """Create a minimal TaskOutput-like object with a .raw attribute."""
    mock = MagicMock()
    mock.raw = raw
    return mock


# ---------------------------------------------------------------------------
# validate_analysis_has_tool_data
# ---------------------------------------------------------------------------

class TestValidateAnalysisHasToolData:

    def test_valid_analysis_passes(self):
        data = {
            "customer_id": "C001",
            "risk_score": 0.4,
            "preliminary_verdict": "APPROVED",
        }
        ok, _ = validate_analysis_has_tool_data(_make_task_output(json.dumps(data)))
        assert ok is True

    def test_missing_customer_id_fails(self):
        data = {"risk_score": 0.4, "preliminary_verdict": "APPROVED"}
        ok, msg = validate_analysis_has_tool_data(_make_task_output(json.dumps(data)))
        assert ok is False
        assert "customer_id" in msg

    def test_placeholder_customer_id_fails(self):
        data = {
            "customer_id": "placeholder",
            "risk_score": 0.4,
            "preliminary_verdict": "APPROVED",
        }
        ok, msg = validate_analysis_has_tool_data(_make_task_output(json.dumps(data)))
        assert ok is False
        assert "placeholder" in msg

    def test_missing_risk_score_fails(self):
        data = {"customer_id": "C001", "preliminary_verdict": "APPROVED"}
        ok, msg = validate_analysis_has_tool_data(_make_task_output(json.dumps(data)))
        assert ok is False
        assert "risk_score" in msg

    def test_invalid_risk_score_fails(self):
        data = {
            "customer_id": "C001",
            "risk_score": 1.5,
            "preliminary_verdict": "APPROVED",
        }
        ok, msg = validate_analysis_has_tool_data(_make_task_output(json.dumps(data)))
        assert ok is False
        assert "valid range" in msg

    def test_invalid_verdict_fails(self):
        data = {
            "customer_id": "C001",
            "risk_score": 0.4,
            "preliminary_verdict": "MAYBE",
        }
        ok, msg = validate_analysis_has_tool_data(_make_task_output(json.dumps(data)))
        assert ok is False
        assert "FRAUD" in msg or "APPROVED" in msg

    def test_non_json_passes_gracefully(self):
        ok, _ = validate_analysis_has_tool_data(_make_task_output("not json at all"))
        assert ok is True


# ---------------------------------------------------------------------------
# validate_verdict_risk_alignment
# ---------------------------------------------------------------------------

class TestValidateVerdictRiskAlignment:

    def test_low_score_approved_passes(self):
        data = {"risk_score": 0.2, "verdict": "APPROVED"}
        ok, _ = validate_verdict_risk_alignment(_make_task_output(json.dumps(data)))
        assert ok is True

    def test_high_score_fraud_passes(self):
        data = {"risk_score": 0.8, "verdict": "FRAUD"}
        ok, _ = validate_verdict_risk_alignment(_make_task_output(json.dumps(data)))
        assert ok is True

    def test_high_score_approved_fails(self):
        data = {"risk_score": 0.8, "verdict": "APPROVED"}
        ok, msg = validate_verdict_risk_alignment(_make_task_output(json.dumps(data)))
        assert ok is False
        assert "Inconsistency" in msg

    def test_low_score_fraud_fails(self):
        data = {"risk_score": 0.2, "verdict": "FRAUD"}
        ok, msg = validate_verdict_risk_alignment(_make_task_output(json.dumps(data)))
        assert ok is False
        assert "Inconsistency" in msg

    def test_invalid_verdict_value_fails(self):
        data = {"risk_score": 0.5, "verdict": "SUSPICIOUS"}
        ok, msg = validate_verdict_risk_alignment(_make_task_output(json.dumps(data)))
        assert ok is False
        assert "FRAUD" in msg or "APPROVED" in msg

    def test_out_of_range_score_fails(self):
        data = {"risk_score": -0.5, "verdict": "APPROVED"}
        ok, msg = validate_verdict_risk_alignment(_make_task_output(json.dumps(data)))
        assert ok is False
        assert "valid range" in msg


# ---------------------------------------------------------------------------
# validate_required_fields
# ---------------------------------------------------------------------------

class TestValidateRequiredFields:

    def test_valid_verdict_passes(self):
        data = {
            "customer_id": "C001",
            "verdict": "APPROVED",
            "risk_score": 0.2,
            "explanation": "The transaction is a small amount to a known merchant with low risk.",
            "transaction_amount": 6.36,
        }
        ok, _ = validate_required_fields(_make_task_output(json.dumps(data)))
        assert ok is True

    def test_missing_fields_fails(self):
        data = {"customer_id": "C001"}
        ok, msg = validate_required_fields(_make_task_output(json.dumps(data)))
        assert ok is False
        assert "Missing required fields" in msg

    def test_short_explanation_fails(self):
        data = {
            "customer_id": "C001",
            "verdict": "APPROVED",
            "risk_score": 0.2,
            "explanation": "Looks fine.",
        }
        ok, msg = validate_required_fields(_make_task_output(json.dumps(data)))
        assert ok is False
        assert "too short" in msg

    def test_zero_amount_fraud_fails(self):
        data = {
            "customer_id": "C001",
            "verdict": "FRAUD",
            "risk_score": 0.8,
            "explanation": "High risk transaction with suspicious patterns detected in the system.",
            "transaction_amount": 0,
        }
        ok, msg = validate_required_fields(_make_task_output(json.dumps(data)))
        assert ok is False
        assert "transaction_amount" in msg
