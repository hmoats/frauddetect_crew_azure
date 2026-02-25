from __future__ import annotations

import json
from typing import Any, List, Tuple

from crewai import Agent, Crew, Process, Task, LLM, TaskOutput
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.knowledge.source.text_file_knowledge_source import TextFileKnowledgeSource
from crewai.project import CrewBase, agent, crew, task
from pydantic import BaseModel, Field

from frauddetect_crew.tools.model_scoring_tool import ModelScoringTool
from frauddetect_crew.tools.transaction_lookup_tool import TransactionLookupTool


# ---------------------------------------------------------------------------
# Pydantic output model
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Guardrail functions
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Crew definition
# ---------------------------------------------------------------------------

@CrewBase
class FraudDetectCrew:
    """Multi-agent fraud detection crew with analyst and validator."""

    agents: List[BaseAgent]
    tasks: List[Task]

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def fraud_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["fraud_analyst"],  # type: ignore[index]
            tools=[TransactionLookupTool(), ModelScoringTool()],
            verbose=True,
            max_iter=15,
            max_rpm=30,
            respect_context_window=True,
            llm=LLM(model="openai/gpt-4o-mini", temperature=0.1),
        )

    @agent
    def verdict_validator(self) -> Agent:
        return Agent(
            config=self.agents_config["verdict_validator"],  # type: ignore[index]
            tools=[TransactionLookupTool(), ModelScoringTool()],
            verbose=True,
            max_iter=10,
            max_rpm=30,
            respect_context_window=True,
            llm=LLM(model="openai/gpt-4o-mini", temperature=0.0),
        )

    @task
    def evaluate_transaction(self) -> Task:
        return Task(
            config=self.tasks_config["evaluate_transaction"],  # type: ignore[index]
        )

    @task
    def validate_verdict(self) -> Task:
        return Task(
            config=self.tasks_config["validate_verdict"],  # type: ignore[index]
            output_pydantic=FraudVerdict,
            guardrails=[
                validate_required_fields,
                validate_verdict_risk_alignment,
            ],
            guardrail_max_retries=3,
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Fraud Detection Crew."""
        fraud_policy = TextFileKnowledgeSource(
            file_paths=["fraud_policy.txt"],
        )

        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            memory=True,
            knowledge_sources=[fraud_policy],
            embedder={
                "provider": "openai",
                "config": {"model_name": "text-embedding-3-small"},
            },
        )
