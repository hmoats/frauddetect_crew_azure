# FraudDetect -- AI-Powered Transaction Fraud Detection

A single-crew fraud detection system built with [CrewAI](https://docs.crewai.com). Given a customer ID, the crew retrieves the transaction record, scores it against a pre-trained RandomForest model, and returns a **FRAUD** or **APPROVED** verdict with a plain-English explanation.

## How It Works

```
Input: customer_id
  │
  ▼
┌─────────────────────────────────────────────┐
│  Senior Fraud Detection Analyst  (Agent)    │
│                                             │
│  1. transaction_lookup  ──► fetch record    │
│  2. fraud_model_scorer  ──► predict + score │
│  3. Analyze & render verdict                │
└─────────────────────────────────────────────┘
  │
  ▼
Output: FraudVerdict (structured JSON)
  ├── verdict: FRAUD | APPROVED
  ├── risk_score: 0.0 – 1.0
  ├── explanation: "The transaction..."
  └── transaction fields (amount, merchant, city, etc.)
```

## CrewAI Concepts Demonstrated

| Concept | Where |
|---|---|
| **@CrewBase decorator** | `crew.py` -- single crew class with YAML-driven config |
| **YAML agent/task config** | `config/agents.yaml`, `config/tasks.yaml` -- role, goal, backstory, and task instructions separated from code |
| **Custom Tools (BaseTool)** | `tools/transaction_lookup_tool.py` -- CSV lookup; `tools/model_scoring_tool.py` -- sklearn model inference |
| **Structured Output (output_pydantic)** | Task returns a typed `FraudVerdict` Pydantic model, not free-form text |
| **LLM Configuration** | Agent uses `LLM(model="openai/gpt-4o-mini", temperature=0.1)` for deterministic, cost-efficient scoring |
| **Tool-grounded reasoning** | Agent must call both tools before rendering a verdict -- no hallucinated data |
| **CrewAI Enterprise deployment** | `main.py` exposes `run()` accepting `customer_id` input via `/kickoff` API |

## Quick Start

```bash
# Prerequisites: Python >=3.10, uv, OpenAI API key
cp .env.example .env   # add OPENAI_API_KEY

crewai install          # install dependencies
crewai run              # run with default demo customer

# Or specify a customer
CUSTOMER_ID="63f582f6-e008-4153-8f65-3cb69bddbd3a" crewai run
```

## Project Structure

```
fraud_detection_crew/
├── src/frauddetect_crew/
│   ├── main.py          # Entry point (run function)
│   ├── crew.py          # @CrewBase crew definition
│   ├── state.py         # FraudVerdict Pydantic output model
│   ├── config/
│   │   ├── agents.yaml  # Agent roles, goals, backstories
│   │   └── tasks.yaml   # Task descriptions + expected outputs
│   └── tools/
│       ├── transaction_lookup_tool.py  # CSV customer lookup
│       └── model_scoring_tool.py       # Model predict + score
├── data/
│   └── transactions.csv # Static transaction records
├── models/
│   ├── best_model.pkl   # Pre-trained RandomForest
│   ├── feature_columns.json
│   └── model_metrics.json
├── scripts/
│   └── train_model.py   # Offline model retraining (no LLM)
└── tests/               # 11 unit tests
```

## Retraining the Model

The model is trained offline -- no LLM or CrewAI dependency:

```bash
python scripts/train_model.py
```

This fetches the dataset from GitHub, trains a RandomForest with GridSearchCV, and writes artifacts to `models/`.

## Tests

```bash
uv run pytest tests/ -v
```
