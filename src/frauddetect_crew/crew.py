from typing import List

from crewai import Agent, Crew, Process, Task, LLM
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task

from frauddetect_crew.state import FraudVerdict
from frauddetect_crew.tools.model_scoring_tool import ModelScoringTool
from frauddetect_crew.tools.transaction_lookup_tool import TransactionLookupTool


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
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Fraud Detection Crew."""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
