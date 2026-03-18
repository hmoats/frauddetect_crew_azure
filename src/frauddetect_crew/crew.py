from __future__ import annotations

import json
import logging
from typing import Any, List

from crewai import Agent, Crew, Process, Task, LLM, TaskOutput
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.knowledge.source.text_file_knowledge_source import TextFileKnowledgeSource
from crewai.project import CrewBase, agent, crew, task
from pydantic import BaseModel, Field

from frauddetect_crew.tools.model_scoring_tool import ModelScoringTool
from frauddetect_crew.tools.transaction_lookup_tool import TransactionLookupTool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic output models
# ---------------------------------------------------------------------------

class PreliminaryAnalysis(BaseModel):
    """Structured output from the analyst's initial evaluation."""

    customer_id: str = Field(description="The Customer_ID that was evaluated.")
    transaction_id: str = Field(default="", description="Transaction_ID from the record.")
    transaction_amount: float = Field(default=0.0, description="Transaction amount.")
    merchant_category: str = Field(default="", description="Merchant category.")
    transaction_type: str = Field(default="", description="Transaction type (Debit, Credit, etc).")
    account_balance: float = Field(default=0.0, description="Account balance at time of transaction.")
    city: str = Field(default="", description="City of the customer.")
    device_type: str = Field(default="", description="Device used for the transaction.")
    risk_score: float = Field(description="Model probability of fraud, 0.0 to 1.0.")
    predicted_label: int = Field(description="Model prediction: 0 = legitimate, 1 = fraud.")
    preliminary_verdict: str = Field(description="FRAUD or APPROVED based on initial analysis.")
    reasoning: str = Field(
        description="Evidence-based reasoning referencing the model score and transaction attributes."
    )


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

def validate_analysis_has_tool_data(result: TaskOutput):
    """Ensure the analyst actually used both tools and returned real data."""
    try:
        data = json.loads(result.raw) if isinstance(result.raw, str) else {}

        # Check that the analyst retrieved a real customer_id (not empty/placeholder)
        cid = data.get("customer_id", "")
        if not cid or cid in ("unknown", "N/A", "placeholder"):
            return (
                False,
                "customer_id is missing or a placeholder. Use the transaction_lookup "
                "tool to retrieve the real customer record before forming your analysis.",
            )

        # Check that a risk_score was returned from the model scorer
        score = data.get("risk_score")
        if score is None:
            return (
                False,
                "risk_score is missing. Use the fraud_model_scorer tool to score "
                "the transaction and include the risk_score in your output.",
            )

        score = float(score)
        if score < 0 or score > 1:
            return (
                False,
                f"risk_score {score} is outside the valid range [0.0, 1.0]. "
                "Re-check the fraud_model_scorer output.",
            )

        # Check that a preliminary verdict was provided
        verdict = data.get("preliminary_verdict", "").upper().strip()
        if verdict not in ("FRAUD", "APPROVED"):
            return (
                False,
                f"preliminary_verdict must be 'FRAUD' or 'APPROVED', got '{verdict}'.",
            )

        return (True, result.raw)
    except (json.JSONDecodeError, ValueError, TypeError):
        return (True, result.raw)


def validate_verdict_risk_alignment(result: TaskOutput):
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


def validate_required_fields(result: TaskOutput):
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
# Callback functions
# ---------------------------------------------------------------------------

def log_verdict_completed(output: TaskOutput) -> None:
    """Log a summary when the final verdict is produced.

    Callbacks run after a task completes successfully (post-guardrail).
    Unlike guardrails, they do not trigger retries -- they are for
    side effects like logging, notifications, or metrics.
    """
    try:
        data = json.loads(output.raw) if isinstance(output.raw, str) else {}
        verdict = data.get("verdict", "UNKNOWN")
        score = data.get("risk_score", "N/A")
        cid = data.get("customer_id", "N/A")
        logger.info(
            "Verdict complete | customer=%s | verdict=%s | risk_score=%s",
            cid, verdict, score,
        )
    except (json.JSONDecodeError, ValueError, TypeError):
        logger.warning("Verdict callback: could not parse output.")


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
            tools=[],
            verbose=True,
            max_iter=10,
            max_rpm=30,
            respect_context_window=True,
            memory=True,
            llm=LLM(model="openai/gpt-4o-mini", temperature=0.0),
        )

    @task
    def evaluate_transaction(self) -> Task:
        return Task(
            config=self.tasks_config["evaluate_transaction"],  # type: ignore[index]
            output_pydantic=PreliminaryAnalysis,
            guardrail=validate_analysis_has_tool_data,
            guardrail_max_retries=3,
        )

    @task
    def validate_verdict(self) -> Task:
        return Task(
            config=self.tasks_config["validate_verdict"],  # type: ignore[index]
            output_pydantic=FraudVerdict,
            output_file="output/verdict.json",
            guardrails=[
                validate_required_fields,
                validate_verdict_risk_alignment,
            ],
            guardrail_max_retries=3,
            callback=log_verdict_completed,
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
