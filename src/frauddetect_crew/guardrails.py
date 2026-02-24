from __future__ import annotations

import json
from typing import Any, Tuple

from crewai import TaskOutput


def validate_verdict_risk_alignment(result: TaskOutput) -> Tuple[bool, Any]:
    """Reject verdicts that contradict the model risk score."""
    try:
        data = json.loads(result.raw) if isinstance(result.raw, str) else {}
        score = float(data.get("risk_score", -1))
        verdict = data.get("verdict", "").upper().strip()

        if score < 0 or score > 1:
            return (
                False,
                f"risk_score {score} is outside the valid range [0.0, 1.0]. "
                "Re-check the model output and correct the score.",
            )

        if verdict not in ("FRAUD", "APPROVED"):
            return (
                False,
                f"verdict must be exactly 'FRAUD' or 'APPROVED', got '{verdict}'.",
            )

        if score > 0.5 and verdict == "APPROVED":
            return (
                False,
                f"Inconsistency: risk_score is {score} (> 0.5) but verdict is "
                "APPROVED. A score above 0.5 should generally be FRAUD. Either "
                "correct the verdict or justify the override in the explanation.",
            )

        if score <= 0.5 and verdict == "FRAUD":
            return (
                False,
                f"Inconsistency: risk_score is {score} (<= 0.5) but verdict is "
                "FRAUD. A score at or below 0.5 should generally be APPROVED. "
                "Either correct the verdict or justify the override in the explanation.",
            )

        return (True, result.raw)
    except (json.JSONDecodeError, ValueError, TypeError):
        return (True, result.raw)


def validate_required_fields(result: TaskOutput) -> Tuple[bool, Any]:
    """Ensure the verdict contains real, populated data rather than placeholders."""
    try:
        data = json.loads(result.raw) if isinstance(result.raw, str) else {}

        required = ["customer_id", "verdict", "risk_score", "explanation"]
        missing = [f for f in required if not data.get(f)]
        if missing:
            return (
                False,
                f"Missing required fields: {', '.join(missing)}. "
                "Populate every field with real data from the transaction record.",
            )

        explanation = str(data.get("explanation", ""))
        if len(explanation.split()) < 8:
            return (
                False,
                "Explanation is too short. Provide a substantive 2-3 sentence "
                "explanation referencing real transaction attributes (amount, "
                "merchant, device, balance).",
            )

        if data.get("transaction_amount", 0) == 0 and data.get("verdict") == "FRAUD":
            return (
                False,
                "transaction_amount is 0 for a FRAUD verdict — this is likely a "
                "missing value. Verify the amount from the transaction record.",
            )

        return (True, result.raw)
    except (json.JSONDecodeError, ValueError, TypeError):
        return (True, result.raw)
