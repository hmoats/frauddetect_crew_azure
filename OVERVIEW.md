# FraudDetect Crew -- Execution Overview

## Business Purpose

Financial institutions process thousands of transaction alerts daily. Traditional manual review costs roughly $45-65 per alert and takes 30-45 minutes of analyst time. This automation evaluates a single transaction against a pre-trained machine learning model and delivers a **FRAUD** or **APPROVED** verdict in under 30 seconds, with a plain-English explanation suitable for downstream case management systems, compliance records, or customer-facing communications.

The system is designed for deployment on CrewAI Enterprise, where it accepts a `customer_id` via API and returns a structured JSON verdict -- making it embeddable in existing payment processing pipelines, alert management platforms, or real-time transaction monitoring workflows.

## What Happens During Execution

The crew runs two AI agents sequentially against a single transaction:

### Agent 1 -- Senior Fraud Detection Analyst

The analyst receives a `customer_id` and executes two tool calls:

1. **Transaction Lookup** -- Retrieves the customer's transaction record from a local database (CSV), returning all fields: amount, merchant category, device type, location, account balance, timestamp, and transaction type.

2. **Model Scoring** -- Passes the transaction record to a pre-trained RandomForest classifier (97.99% accuracy, 0.98 F1 score, trained on 2,740 labeled transactions via 5-fold cross-validated GridSearchCV). The model encodes the record into 9,619 features via one-hot encoding and returns a fraud probability score (0.0--1.0) along with the binary prediction.

The analyst then contextualizes the model score against the transaction attributes -- a high score on a small transaction to a known merchant is treated differently than the same score on a large transaction to an unknown merchant at 3am with a zero account balance. The agent produces a structured `PreliminaryAnalysis` with the transaction data, model results, and evidence-based reasoning.

**Guardrail:** A function-based guardrail (`validate_analysis_has_tool_data`) verifies the analyst actually called both tools and returned real data -- checking for a valid customer_id, a risk_score in the correct range, and a valid preliminary verdict. If the check fails, the agent retries up to 3 times.

### Agent 2 -- Verdict Quality Assurance Specialist

The validator receives the analyst's structured evaluation (via `context` dependency) and performs an independent quality check:

- **Score-verdict alignment** -- Does a 0.4 risk score actually justify FRAUD, or should it be APPROVED?
- **Evidence grounding** -- Does the explanation reference real transaction data, or is it generic boilerplate?
- **Red flag coverage** -- Were obvious indicators missed (zero balance, unknown merchant, amount exceeding balance)?
- **Data integrity** -- Are all output fields populated with values from the actual record?

The validator has access to the same tools and can independently re-score the transaction if the analyst's data looks inconsistent. If the verdict is sound, it is confirmed. If not, the validator corrects it and documents what changed.

**Guardrails:** Two chained function-based guardrails run sequentially on the final output:
1. `validate_required_fields` -- Ensures all required fields are populated and the explanation is substantive.
2. `validate_verdict_risk_alignment` -- Rejects verdicts where the risk score and FRAUD/APPROVED label contradict each other.

**Callback:** After successful validation, `log_verdict_completed` logs a summary of the verdict for monitoring and audit purposes. Unlike guardrails, callbacks do not trigger retries -- they are for side effects.

**Output file:** The final verdict is automatically written to `output/verdict.json` in addition to being returned as a structured object.

## Output

The crew returns a structured `FraudVerdict` JSON object:

```json
{
  "customer_id": "d56303bd-a099-4650-bfcb-61d047b74e13",
  "verdict": "APPROVED",
  "risk_score": 0.4,
  "explanation": "The transaction amount of 6.36 is well within the account balance of 24444.05, and the merchant category is not flagged as suspicious. The model's prediction supports this conclusion.",
  "transaction_id": "dc1eac53-f887-4057-9dfc-e24bfbd346fe",
  "transaction_amount": 6.36,
  "merchant_category": "Digital Services",
  "transaction_type": "Bill Payment",
  "account_balance": 24444.05,
  "city": "Mumbai",
  "device_type": "Mobile"
}
```

## CrewAI Concepts Demonstrated

| Concept | Where in Code |
|---|---|
| `@CrewBase` crew pattern | `crew.py` -- class decorator, `@agent`, `@task`, `@crew` |
| YAML-first config | `config/agents.yaml`, `config/tasks.yaml` |
| Variable interpolation | `{customer_id}` in task description, passed via `kickoff(inputs=)` |
| Custom tools (`BaseTool`) | `tools/transaction_lookup_tool.py`, `tools/model_scoring_tool.py` |
| Pydantic input schemas | `TransactionLookupInput`, `ModelScoringInput` on each tool |
| Structured output (`output_pydantic`) | `PreliminaryAnalysis` on task 1, `FraudVerdict` on task 2 |
| Function-based guardrails | `validate_analysis_has_tool_data` (task 1), chained guardrails (task 2) |
| Guardrail chaining | `guardrails=[validate_required_fields, validate_verdict_risk_alignment]` |
| Guardrail retries | `guardrail_max_retries=3` on both tasks |
| Task context / dependencies | `context: [evaluate_transaction]` on the validator task |
| Output file | `output_file="output/verdict.json"` on the final task |
| Callback | `log_verdict_completed` on the final task |
| Knowledge system | `fraud_policy.txt` loaded as `TextFileKnowledgeSource` at crew level |
| Memory | `memory=True` on crew and agents, custom OpenAI embedder |
| Agent parameters | `max_iter`, `max_rpm`, `respect_context_window`, per-agent LLM + temperature |
| Sequential process | `Process.sequential` with 2 agents, 2 tasks |
| Test suite | `tests/` -- Pydantic models, tools, crew config, guardrails |

## Technical Stack

| Component | Technology |
|---|---|
| Orchestration | CrewAI 1.9.3 (`@CrewBase`, sequential process) |
| Agents | 2 agents, both on `gpt-4o-mini` (low cost, low latency) |
| ML Model | scikit-learn RandomForest (100 estimators, GridSearchCV-tuned) |
| Tools | 2 custom `BaseTool` implementations (CSV lookup, sklearn inference) |
| Output | Pydantic `output_pydantic` on both tasks for typed structured responses |
| Guardrails | 3 function-based guardrails across 2 tasks |
| Knowledge | Text file knowledge source (fraud policy) |
| Deployment | CrewAI Enterprise `/kickoff` API with `customer_id` input |
