# Configuration

This folder holds Kedro configuration.

| Env | Path | Loaded when |
|---|---|---|
| `base` | `conf/base/` | Always — `catalog.yml` (SQL, ML model, LLM, prompts), `parameters.yml`, `credentials.yml.template` |
| `local` | `conf/local/` | Default run env; empty; gitignored |
| `langfuse` | `conf/langfuse/` | `--env langfuse` — `catalog_genai_config.yml` (Langfuse `intent_prompt`, `intent_tracer`), `catalog_evaluation.yml` |

Copy `base/credentials.yml.template` to `local/credentials.yml` and `langfuse/credentials.yml` (gitignored). Kedro loads `base` plus one run env, so `--env langfuse` does not read `local`: each run env carries its own credentials. It reads `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and `LANGFUSE_*` environment variables. `default_run_env = "local"` and `base_env = "base"` in `settings.py`, so `kedro run` loads `base` + `local`; add `--env langfuse` to layer on evaluation datasets for `intent_detection_evaluation` and `extraction_evaluation`.