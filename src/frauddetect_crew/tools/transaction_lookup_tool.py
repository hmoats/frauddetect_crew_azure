import json
import os
from typing import Type

import pandas as pd
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


class TransactionLookupInput(BaseModel):
    """Input schema for the transaction lookup tool."""

    customer_id: str = Field(..., description="The Customer_ID to look up.")


class TransactionLookupTool(BaseTool):
    """Looks up a customer's transaction record from the static transactions CSV."""

    name: str = "transaction_lookup"
    description: str = (
        "Looks up a customer transaction record by Customer_ID from the local "
        "transactions database (CSV). Returns the full transaction record as JSON "
        "including amount, merchant, device, location, and account balance. "
        "Also returns a list of all available Customer_IDs if the lookup fails."
    )
    args_schema: Type[BaseModel] = TransactionLookupInput

    def _run(self, customer_id: str) -> str:
        csv_path = os.path.join(DATA_DIR, "transactions.csv")
        if not os.path.exists(csv_path):
            return json.dumps({
                "status": "error",
                "message": f"Transactions file not found at {csv_path}",
            })

        df = pd.read_csv(csv_path, dtype=str)
        matches = df[df["Customer_ID"] == customer_id]

        if matches.empty:
            available = df["Customer_ID"].unique().tolist()
            return json.dumps({
                "status": "not_found",
                "message": f"Customer_ID '{customer_id}' not found.",
                "available_ids": available,
            })

        record = matches.iloc[0].where(matches.iloc[0].notna(), "").to_dict()

        numeric_fields = ["Age", "Transaction_Amount", "Account_Balance"]
        for field in numeric_fields:
            if field in record and record[field]:
                try:
                    record[field] = float(record[field])
                except ValueError:
                    pass

        return json.dumps({"status": "found", "record": record}, indent=2)
