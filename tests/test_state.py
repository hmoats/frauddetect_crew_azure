"""Tests for the Pydantic output models."""

from frauddetect_crew.crew import FraudVerdict, PreliminaryAnalysis


# ---------------------------------------------------------------------------
# PreliminaryAnalysis tests
# ---------------------------------------------------------------------------

def test_preliminary_analysis_required_fields():
    analysis = PreliminaryAnalysis(
        customer_id="C001",
        risk_score=0.75,
        predicted_label=1,
        preliminary_verdict="FRAUD",
        reasoning="High risk score with suspicious merchant category.",
    )
    assert analysis.customer_id == "C001"
    assert analysis.risk_score == 0.75
    assert analysis.predicted_label == 1
    assert analysis.preliminary_verdict == "FRAUD"
    # Defaults
    assert analysis.transaction_id == ""
    assert analysis.transaction_amount == 0.0


def test_preliminary_analysis_full():
    analysis = PreliminaryAnalysis(
        customer_id="C002",
        transaction_id="TXN-456",
        transaction_amount=112236.14,
        merchant_category="Unknown Merchant",
        transaction_type="Bill Payment",
        account_balance=0.0,
        city="Mumbai",
        device_type="Mobile",
        risk_score=0.92,
        predicted_label=1,
        preliminary_verdict="FRAUD",
        reasoning="Zero balance, unknown merchant, and high model score.",
    )
    assert analysis.transaction_amount == 112236.14
    assert analysis.account_balance == 0.0
    assert analysis.merchant_category == "Unknown Merchant"


def test_preliminary_analysis_serialization():
    analysis = PreliminaryAnalysis(
        customer_id="C003",
        risk_score=0.15,
        predicted_label=0,
        preliminary_verdict="APPROVED",
        reasoning="Low risk transaction to known merchant.",
    )
    data = analysis.model_dump()
    assert data["customer_id"] == "C003"
    assert data["preliminary_verdict"] == "APPROVED"
    assert isinstance(data["risk_score"], float)
    assert isinstance(data["predicted_label"], int)


# ---------------------------------------------------------------------------
# FraudVerdict tests
# ---------------------------------------------------------------------------

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
