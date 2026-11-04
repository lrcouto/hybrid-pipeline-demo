# Demo script (7 minutes)

Hybrid ML + agentic pipeline in Kedro. Audience: engineers and data scientists who want to see code.

Two ideas to land:
1. `llm_context_node`: the LLM, prompts and tools are declared in the pipeline and supplied by the catalog, so node functions stay plain Python.
2. One catalog for everything: SQL tables, the sklearn model, the LLM and the prompts are all just datasets.

Spoken lines are a guide, not something to memorise. Times are where you should be when the beat ends.

---

## Before you start (15 minutes before)

```bash
cd hybrid-pipeline-demo
source .venv/bin/activate
python create_db_and_data.py          # resets the DB; every practice run adds claims to it
kedro run --pipelines triage_training # only if data/06_models/triage_model.pkl is missing
kedro run                             # warm-up: proves the key works, catches network problems
```

Then open, left to right:

| Window | What |
|---|---|
| Browser tab 1 | `kedro viz run` (default pipeline selected, styles applied) |
| Browser tab 2 | Langfuse, logged in, on the `extraction` experiment comparison |
| Editor | `conf/base/catalog.yml`, `claim_scoring/pipeline.py`, `claim_scoring/nodes.py`, `conf/langfuse/catalog_genai_config.yml` as tabs in that order |
| Terminal 1 | `kedro server start --port 8765` running |
| Terminal 2 | in the repo root, empty, font size up |

Checks: editor and terminal font big enough for the back of the room, notifications off, Langfuse experiment already run at least once so the page has data.

---

## 0:00 – 0:40 · The problem, and the picture

**Show:** Kedro-Viz, the default pipeline.

> A claim comes in as free text. A customer writes "someone rear-ended me, my neck hurts, the shop quoted 18 grand". We have a classic ML model that decides whether a claim needs a human adjuster, but the model wants numbers and booleans, not a paragraph.
>
> So the pipeline is: an LLM reads the message and fills in the model's features, the model scores it, and another LLM writes the reply.
>
> Blue nodes are agentic, yellow nodes are classic ML, grey is the database. It's one pipeline, and Kedro doesn't treat the two kinds of node differently.

**Point at:** `extraction_context_node` → `extract_claim_features_node` → `predict_review_risk_node` → response agent.

---

## 0:40 – 2:00 · One catalog

**Show:** `conf/base/catalog.yml`. Scroll slowly through the four sections.

> This is the whole data layer. Up top, the SQL tables: user profile, docs, claim history. The user query takes the user ID from the request at runtime.
>
> Then the ML model, which is a pickle file, and its metrics.
>
> Then the GenAI part. The LLM is a dataset. The prompts are datasets, versioned files on disk. Nothing special.
>
> Nodes never import a client or read a file. They ask for `llm` or `triage_model` by name and get it.

**Then:** switch to `conf/langfuse/catalog_genai_config.yml`.

> Separation of concerns: this is a second environment. Same dataset names, but now the intent prompt comes versioned from Langfuse and every LLM call is traced. You switch with `--env langfuse`. No node code changes.

Cut first if you're running late: the Langfuse env file. Say it in one sentence instead.

---

## 2:00 – 3:45 · The LLM context node (the main part)

**Show:** `src/kedro_agentic_workflows/pipelines/claim_scoring/pipeline.py`, lines 10–15.

> This is the new piece. `llm_context_node` says: this step needs the `llm` and the `extraction_prompt`. Both are catalog entries. Kedro loads them and bundles them into one `LLMContext` object.

**Show:** `claim_scoring/nodes.py`, lines 9–26.

> And here's the node that uses it. A Pydantic model describing exactly what the ML model expects: incident type, amount, injuries, police report, third party, days to report.
>
> The function takes the context, asks the LLM for structured output against that schema, and returns a dict. That's the agentic step, about four lines.

**Show:** lines 29–35.

> If the message isn't a new claim, it returns empty and the model step is skipped.

**Optional, if they look interested:** flip to `response_generation/pipeline.py` lines 16–27.

> Same idea with tools: the response agent gets database lookups as tools, and the tools' inputs come from the catalog too.

**Land it:**

> Because the LLM is an input, you can test these functions with a fake LLM, swap models with a runtime parameter, and see every dependency in Viz.

---

## 3:45 – 4:15 · The ML node

**Show:** `claim_scoring/nodes.py`, lines 38–49.

> And the ML side is plain scikit-learn. `predict_proba` on a dataframe. No LLM code anywhere in this function. It was trained by a normal Kedro pipeline, `triage_training`, with a gradient boosting classifier, on 600 synthetic past claims.

If asked about the numbers: accuracy 0.77 and AUC 0.73 on synthetic data. It's there to show the mechanics, not to be a real model.

---

## 4:15 – 5:30 · Live: trigger it through the API

**Show:** Terminal 2. Before running it, show `truck.json` quickly (`cat truck.json`).

> Kedro has a built-in HTTP server. I'm sending the claim as a runtime parameter.

```bash
curl -s -X POST localhost:8765/run -H 'Content-Type: application/json' -d @truck.json
cat data/08_reporting/triage_report.json
```

> In the report you can see the extracted features (auto collision, 18,000, injuries true, no police report, 21 days), then the review probability, high, so it's escalated, and the reply tells the customer an adjuster will review it.

**Then the contrast:**

```bash
curl -s -X POST localhost:8765/run -H 'Content-Type: application/json' -d @breakin.json
cat data/08_reporting/triage_report.json
```

> Car break-in, $3,200, police report filed. Low probability, fast-tracked. Same pipeline, and the model made the call, not the LLM.

The run takes a few seconds per request with gpt-4o. Talk while it runs. Don't read out exact probabilities in advance; they vary a little between runs.

---

## 5:30 – 6:40 · Evaluation

**Show:** Langfuse tab, extraction experiment.

> Because the LLM step has a strict output schema, we can evaluate it like normal code: twelve labelled claims, and we check each field. Exact match on type and booleans, amount within 10%, days within one.

```bash
kedro run --pipelines extraction_evaluation --env langfuse
```

(Run it only if there's time. Otherwise show the result that's already there.)

> Here's an interesting miss: "my phone fell in the lake". The label says `other`, the LLM says `water_damage`. You could argue either way, and that's exactly what you want an eval to surface. You fix it in the prompt, run again, and compare the two runs side by side.
>
> The eval is just another Kedro pipeline, reusing the same `llm_context_node` and the same catalog entries as production.

---

## 6:40 – 7:00 · Wrap up

> So: one catalog for the data store, the model and the LLM pieces. LLM context nodes keep the agentic code small and testable. Classic ML sits in the same pipeline, untouched. And evaluation is a pipeline too.
>
> The code is in the repo, happy to take questions.

---

## If something breaks

| Problem | Do this |
|---|---|
| API call hangs or the LLM errors | Say "network". Show `git show HEAD:data/08_reporting/triage_report.json` (a committed break-in result), then talk through it |
| `{"detail": ... "Field required"}` | The body didn't reach the server. Use `-d @truck.json`, never a pasted multi-line command |
| Langfuse won't load | Skip the live eval run and describe it with the per-field scores |
| Viz won't start | `kedro viz build` beforehand and open `build/index.html` |
| Running over time | Drop the Langfuse env file (0:40 beat) and the response-agent tools aside (2:00 beat) |

## Likely questions

- **Why not just let the LLM decide escalation?** The model is trained on history and auditable. The LLM only translates text into features, which you can evaluate field by field.
- **Can I use another provider?** Yes: any OpenAI-compatible endpoint via `OPENAI_BASE_URL`, and the model name as a runtime parameter. A different LangChain dataset type also works with no node changes.
- **What's `LLMContext`?** A container Kedro builds from the catalog: `.llm`, `.prompts[...]`, `.tools`. Kedro 1.2 and later.
- **Is the server production-ready?** It has no authentication, so bind to localhost or put it behind your own gateway. It returns run status; results go wherever the catalog says.
