from __future__ import annotations

from pydantic import BaseModel, Field


class FraudVerdict(BaseModel):
    """Structured output returned to CrewAI Enterprise for every transaction."""

    customer_id: str = Field(description="The Customer_ID that was evaluated.")
    verdict: str = Field(description="FRAUD or APPROVED.")
    risk_score: float = Field(description="Model probability of fraud, 0.0 to 1.0.")
    explanation: str = Field(
        description="Two to three sentence plain-English explanation of the verdict."
    )
    transaction_id: str = Field(default="", description="Transaction_ID from the record.")
    transaction_amount: float = Field(default=0.0, description="Transaction amount.")
    merchant_category: str = Field(default="", description="Merchant category.")
    transaction_type: str = Field(default="", description="Transaction type (Debit, Credit, etc).")
    account_balance: float = Field(default=0.0, description="Account balance at time of transaction.")
    city: str = Field(default="", description="City of the customer.")
    device_type: str = Field(default="", description="Device used for the transaction.")
