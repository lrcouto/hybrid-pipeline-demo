# Kedro hybrid ML + agentic pipeline demo

A pruned version of [kedro-academy/kedro-agentic-workflows](https://github.com/kedro-org/kedro-academy/tree/main/kedro-agentic-workflows) that mixes LLM agent steps with a classic scikit-learn model in one Kedro project. Insurance claims use case: a customer message comes in over HTTP, an LLM turns the free text into structured features, a pre-trained model predicts whether the claim needs manual review, and an LLM agent writes the reply.

## The flow

```
POST /run {params: {user_id, message}}            kedro server
  -> user profile, docs, claims from SQLite         SQL datasets
  -> intent_detection      llm_context_node + LangGraph agent   (agentic)
  -> claim_scoring         llm_context_node: free text -> features  (agentic)
                           scikit-learn model -> review probability  (ML)
  -> response_generation   llm_context_node + tools + LangGraph     (agentic)
  -> data/08_reporting/triage_report.json, message log in SQLite
```

## Pipelines

| Pipeline | What it does |
|---|---|
| triage_training | Classic ML: split claim_history, train GradientBoostingClassifier in an sklearn Pipeline with OneHotEncoder, evaluate -> triage_model.pkl, triage_metrics.json |
| __default__ | intent_detection + claim_scoring + response_generation (handle one message) |
| extraction_evaluation | Langfuse experiment: 12 labelled claims, deterministic per-field scores, --env langfuse |
| intent_detection_evaluation | Langfuse experiment for intent classifier with an LLM judge, --env langfuse |

## One catalog

`conf/base/catalog.yml` holds the SQL tables, the pickled model, the LLM (`langchain.ChatOpenAIDataset`) and the prompts (`PromptDataset`) side by side. Nodes receive them as inputs and never know where they come from. LLM context nodes (`llm_context_node` from kedro.pipeline) bundle an LLM, prompts and tools into one LLMContext from catalog entries. Requires Kedro >= 1.2; this project pins 1.7.0.

## Config environments

- `base`: no tracing, no Langfuse keys needed. `intent_tracer` is a `NoTracerDataset` that loads `None`.
- `--env langfuse`: the same dataset names, backed by Langfuse: a versioned intent prompt, a LangChain tracing callback and the evaluation datasets. No node code changes.

## Setup

```bash
uv venv -p 3.12 .venv && uv pip install -p .venv/bin/python -r requirements.txt
source .venv/bin/activate
python create_db_and_data.py      # SQLite db: users, claims, docs, 600 synthetic past claims
cp conf/base/credentials.yml.template conf/base/credentials.yml
export OPENAI_API_KEY=...         # any OpenAI-compatible endpoint: also set OPENAI_BASE_URL
kedro run --pipelines triage_training
```

Note: credentials.yml reads keys from environment variables; it is gitignored.

## Run

```bash
kedro run                                                          # default message param
kedro run --params "user_id=3,message=My kitchen flooded yesterday"
```

`--params` splits on commas, so messages with commas need the HTTP server:

```bash
kedro server start --port 8765
# in another terminal:
curl -s -X POST localhost:8765/run -H 'Content-Type: application/json' -d @truck.json    # escalated to an adjuster
curl -s -X POST localhost:8765/run -H 'Content-Type: application/json' -d @breakin.json  # fast-track
cat data/08_reporting/triage_report.json
```

The server returns run status only (status, run_id, duration_ms); results go where the catalog says. The server has no authentication: bind to localhost.

Other models: `--params model_name=<model>`. Some providers only accept temperature 1 (for example Moonshot Kimi): add `temperature=1`.

## Verified results

From runs on 2026-11-04 with Moonshot kimi-k2.7-code-highspeed; gpt-4o is the default and has not been run yet.

- triage_training: accuracy 0.773, ROC AUC 0.729 on 150 held-out synthetic claims.
- Car break-in, $3,200, police report: theft features extracted, review probability 0.048, fast-track reply.
- Truck rear-ended, neck injury, $18,000, 21 days, no police: review probability 0.891, escalated to an adjuster.
- General question: intent general_question, extraction and model skipped, answer from docs lookup.
- Extraction check on the 12 labelled claims (local, without Langfuse): 11/12 fully correct; the miss was "phone fell into the lake" labelled water_damage instead of other.
- Langfuse experiments have not been run (no Langfuse keys yet).

## Tests

`PYTHONPATH=src pytest tests`: 4 unit tests on node functions. No LLM or Kedro session needed.

## What was removed from the original

OpenAI Agents SDK and AutoGen variants, Opik env and evaluation pipeline, interactive questionary input and clarification loop (the message is now a runtime parameter).

## Limits

Synthetic training data with a hand-made logistic rule, so the model's numbers show the mechanics, not real-world accuracy. Escalation advice is from a demo model.