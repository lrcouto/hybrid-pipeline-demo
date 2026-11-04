"""The hybrid pipeline's logic is plain functions: test them without Kedro or an LLM."""

import pandas as pd
import pytest

from kedro_agentic_workflows.pipelines.claim_scoring.nodes import (
    extract_claim_features,
    predict_review_risk,
)
from kedro_agentic_workflows.pipelines.extraction_evaluation.nodes import score_fields
from kedro_agentic_workflows.pipelines.triage_training.nodes import train_model

PARAMS = {
    "features": {
        "categorical": ["incident_type"],
        "numeric": ["estimated_amount", "injuries", "police_report", "third_party", "days_to_report"],
    },
    "random_state": 0,
}

CLAIM = {
    "incident_type": "theft",
    "estimated_amount": 900.0,
    "injuries": False,
    "police_report": True,
    "third_party": False,
    "days_to_report": 0,
}


@pytest.fixture
def model():
    X = pd.DataFrame([CLAIM, {**CLAIM, "estimated_amount": 20000.0, "injuries": True}] * 10)
    y = pd.Series([0, 1] * 10)
    return train_model(X, y, PARAMS)


def test_high_risk_claim_needs_review(model):
    risky = {**CLAIM, "estimated_amount": 20000.0, "injuries": True}
    assert predict_review_risk(model, risky, review_threshold=0.5)["needs_review"]


def test_low_risk_claim_is_fast_tracked(model):
    assert not predict_review_risk(model, CLAIM, review_threshold=0.5)["needs_review"]


def test_non_claim_messages_skip_the_llm_and_the_model(model):
    # No LLM context needed: the intent short-circuits before any call.
    assert extract_claim_features(None, "How do I file a claim?", {"intent": "general_question"}) == {}
    assert predict_review_risk(model, {}, review_threshold=0.5) == {}


def test_score_fields_tolerates_small_numeric_differences():
    output = {**CLAIM, "estimated_amount": 950.0, "days_to_report": 1}
    assert all(score_fields(output, CLAIM).values())
