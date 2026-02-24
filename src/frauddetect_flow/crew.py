from typing import List

from crewai import Agent, Crew, Process, Task, LLM
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task

from frauddetect_flow.state import FraudVerdict
from frauddetect_flow.tools.model_scoring_tool import ModelScoringTool
from frauddetect_flow.tools.transaction_lookup_tool import TransactionLookupTool


@CrewBase
class FraudDetectCrew:
    """Single-transaction fraud detection crew."""

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

    @task
    def evaluate_transaction(self) -> Task:
        return Task(
            config=self.tasks_config["evaluate_transaction"],  # type: ignore[index]
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
