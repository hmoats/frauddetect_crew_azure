#!/usr/bin/env python
"""Fraud detection crew -- single-transaction evaluation.

Accepts a customer_id, looks up the transaction, scores it against a
pre-trained model, and returns a FRAUD / APPROVED verdict.
"""

import os

from dotenv import load_dotenv

load_dotenv()

from frauddetect_flow.crew import FraudDetectCrew


def run():
    """Entry point for ``crewai run``.

    Provide a customer_id via:
      1. CUSTOMER_ID environment variable, or
      2. Default demo ID from the sample data.
    """
    customer_id = os.environ.get(
        "CUSTOMER_ID",
        "d56303bd-a099-4650-bfcb-61d047b74e13",
    )

    inputs = {"customer_id": customer_id}
    result = FraudDetectCrew().crew().kickoff(inputs=inputs)
    print(result.pydantic.model_dump_json(indent=2) if result.pydantic else result.raw)


if __name__ == "__main__":
    run()
