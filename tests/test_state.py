"""Tests for the FraudVerdict output model."""

from frauddetect_crew.state import FraudVerdict


def test_fraud_verdict_defaults():
    verdict = FraudVerdict(
        customer_id="C001",
        verdict="FRAUD",
        risk_score=0.92,
        explanation="High risk transaction.",
    )
    assert verdict.customer_id == "C001"
    assert verdict.verdict == "FRAUD"
    assert verdict.risk_score == 0.92
    assert verdict.transaction_id == ""
    assert verdict.transaction_amount == 0.0


def test_fraud_verdict_full():
    verdict = FraudVerdict(
        customer_id="C002",
        verdict="APPROVED",
        risk_score=0.05,
        explanation="Normal transaction pattern.",
        transaction_id="TXN-123",
        transaction_amount=500.0,
        merchant_category="Groceries",
        transaction_type="Debit",
        account_balance=10000.0,
        city="Mumbai",
        device_type="Mobile",
    )
    assert verdict.verdict == "APPROVED"
    assert verdict.transaction_amount == 500.0
    assert verdict.city == "Mumbai"


def test_fraud_verdict_serialization():
    verdict = FraudVerdict(
        customer_id="C003",
        verdict="FRAUD",
        risk_score=0.88,
        explanation="Suspicious activity detected.",
    )
    data = verdict.model_dump()
    assert data["customer_id"] == "C003"
    assert data["verdict"] == "FRAUD"
    assert isinstance(data["risk_score"], float)
