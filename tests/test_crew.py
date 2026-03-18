"""Smoke tests for the crew configuration."""

from frauddetect_crew.crew import FraudDetectCrew
from frauddetect_crew.crew import FraudVerdict, PreliminaryAnalysis


def test_crew_instantiation():
    """Verify the crew can be built without errors."""
    crew_instance = FraudDetectCrew().crew()
    assert crew_instance is not None
    assert len(crew_instance.agents) == 2
    assert len(crew_instance.tasks) == 2


def test_fraud_analyst_has_tools():
    """Verify the fraud analyst agent has both tools."""
    crew_obj = FraudDetectCrew()
    agent = crew_obj.fraud_analyst()
    tool_names = [t.name for t in agent.tools]
    assert "transaction_lookup" in tool_names
    assert "fraud_model_scorer" in tool_names


def test_validator_has_tools():
    """Verify the validator agent has both tools."""
    crew_obj = FraudDetectCrew()
    agent = crew_obj.verdict_validator()
    tool_names = [t.name for t in agent.tools]
    assert "transaction_lookup" in tool_names
    assert "fraud_model_scorer" in tool_names


def test_evaluate_task_has_pydantic_output():
    """Verify the evaluation task produces PreliminaryAnalysis output."""
    crew_obj = FraudDetectCrew()
    task = crew_obj.evaluate_transaction()
    assert task.output_pydantic == PreliminaryAnalysis


def test_validate_task_has_pydantic_output():
    """Verify the validation task produces FraudVerdict output."""
    crew_obj = FraudDetectCrew()
    task = crew_obj.validate_verdict()
    assert task.output_pydantic == FraudVerdict


def test_validate_task_has_output_file():
    """Verify the validation task writes output to a file."""
    crew_obj = FraudDetectCrew()
    task = crew_obj.validate_verdict()
    assert task.output_file == "output/verdict.json"


def test_evaluate_task_has_guardrail():
    """Verify the evaluation task has a guardrail attached."""
    crew_obj = FraudDetectCrew()
    task = crew_obj.evaluate_transaction()
    assert task.guardrail is not None


def test_validate_task_has_guardrails():
    """Verify the validation task has multiple guardrails attached."""
    crew_obj = FraudDetectCrew()
    task = crew_obj.validate_verdict()
    assert task.guardrails is not None
    assert len(task.guardrails) == 2


def test_validate_task_has_callback():
    """Verify the validation task has a callback attached."""
    crew_obj = FraudDetectCrew()
    task = crew_obj.validate_verdict()
    assert task.callback is not None
