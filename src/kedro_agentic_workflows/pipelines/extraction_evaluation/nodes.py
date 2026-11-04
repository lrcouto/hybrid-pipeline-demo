import logging
from statistics import mean
from typing import Any, Callable

from kedro.pipeline import LLMContext
from langfuse import Evaluation
from langfuse._client.datasets import DatasetClient

from ..claim_scoring.nodes import extract_features

logger = logging.getLogger(__name__)

EXACT_FIELDS = ["incident_type", "injuries", "police_report", "third_party"]


def make_extraction_task(extraction_context: LLMContext) -> Callable[..., dict]:
    """The thing under test: the same extraction the live pipeline runs."""

    def extraction_task(*, item, **kwargs) -> dict:
        return extract_features(extraction_context, item.input["message"])

    return extraction_task


def score_fields(output: dict, expected: dict) -> dict[str, int]:
    """Deterministic, per-field scores. No LLM judge needed: the output is structured."""
    scores = {f: int(output.get(f) == expected[f]) for f in EXACT_FIELDS}
    amount = output.get("estimated_amount") or 0
    scores["estimated_amount"] = int(abs(amount - expected["estimated_amount"]) <= 0.1 * expected["estimated_amount"])
    scores["days_to_report"] = int(abs((output.get("days_to_report") or 0) - expected["days_to_report"]) <= 1)
    return scores


def init_field_evaluator() -> Callable[..., list[Evaluation]]:
    def field_evaluator(*, output: dict, expected_output: dict, **kwargs) -> list[Evaluation]:
        scores = score_fields(output, expected_output)
        return [Evaluation(name=field, value=value) for field, value in scores.items()] + [
            Evaluation(name="all_fields_correct", value=int(all(scores.values())))
        ]

    return field_evaluator


def run_extraction_experiment(
    extraction_evaluation_data: DatasetClient,
    extraction_task: Callable,
    field_evaluator: Callable,
    model_name: str,
) -> dict[str, Any]:
    """Run the experiment in Langfuse and keep a local summary."""
    result = extraction_evaluation_data.run_experiment(
        name=f"extraction_{model_name}",
        task=extraction_task,
        evaluators=[field_evaluator],
        metadata={"model_name": model_name},
    )
    per_metric: dict[str, list[float]] = {}
    for item in result.item_results:
        for evaluation in item.evaluations:
            per_metric.setdefault(evaluation.name, []).append(evaluation.value)

    summary = {name: round(mean(values), 3) for name, values in per_metric.items()}
    logger.info("Extraction experiment '%s': %s", result.run_name, summary)
    return {"model_name": model_name, "items": len(result.item_results), "accuracy": summary}
