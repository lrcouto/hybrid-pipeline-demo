"""Project pipelines."""

from __future__ import annotations

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    - ``triage_training``: train the claim triage model (classic ML).
    - ``__default__``: handle one customer message end to end
      (agentic → ML → agentic).
    - ``intent_detection_evaluation`` / ``extraction_evaluation``: Langfuse
      experiments, run with ``--env langfuse``.
    """
    pipelines = find_pipelines()
    # `kedro run --pipelines X` only imports X, so build the default only when
    # all of its parts were loaded.
    handle_message = ["intent_detection", "claim_scoring", "response_generation"]
    if all(name in pipelines for name in handle_message):
        pipelines["__default__"] = sum(pipelines[name] for name in handle_message)
    return pipelines
