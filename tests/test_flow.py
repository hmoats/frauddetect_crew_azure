"""Smoke tests for the crew configuration."""

from frauddetect_flow.crew import FraudDetectCrew
from frauddetect_flow.state import FraudVerdict


def test_crew_instantiation():
    """Verify the crew can be built without errors."""
    crew_instance = FraudDetectCrew().crew()
    assert crew_instance is not None
    assert len(crew_instance.agents) == 1
    assert len(crew_instance.tasks) == 1


def test_agent_has_tools():
    """Verify the fraud analyst agent has both tools."""
    crew_obj = FraudDetectCrew()
    agent = crew_obj.fraud_analyst()
    tool_names = [t.name for t in agent.tools]
    assert "transaction_lookup" in tool_names
    assert "fraud_model_scorer" in tool_names


def test_task_has_pydantic_output():
    """Verify the task is configured with FraudVerdict output."""
    crew_obj = FraudDetectCrew()
    task = crew_obj.evaluate_transaction()
    assert task.output_pydantic == FraudVerdict
