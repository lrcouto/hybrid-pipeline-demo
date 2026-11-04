from typing import Literal

import pandas as pd
from kedro.pipeline import LLMContext
from pydantic import BaseModel, Field
from sklearn.pipeline import Pipeline


class ClaimFeatures(BaseModel):
    """The structured input the triage model expects."""

    incident_type: Literal[
        "auto_collision", "theft", "fire", "water_damage", "weather", "other"
    ]
    estimated_amount: float = Field(description="Total loss in US dollars")
    injuries: bool
    police_report: bool
    third_party: bool
    days_to_report: int = Field(description="Days between the incident and the message")


def extract_features(extraction_context: LLMContext, message: str) -> dict:
    """Ask the LLM for the features, validated against ``ClaimFeatures``."""
    llm = extraction_context.llm.with_structured_output(ClaimFeatures)
    chain = extraction_context.prompts["extraction_prompt"] | llm
    return chain.invoke({"message": message}).model_dump()


def extract_claim_features(
    extraction_context: LLMContext, message: str, intent_detection_result: dict
) -> dict:
    """Agentic step: turn the free-text claim into model features."""
    if intent_detection_result["intent"] != "claim_new":
        return {}  # nothing to score
    return extract_features(extraction_context, message)


def predict_review_risk(
    triage_model: Pipeline, claim_features: dict, review_threshold: float
) -> dict:
    """ML step: plain scikit-learn inference on the extracted features."""
    if not claim_features:
        return {}

    probability = triage_model.predict_proba(pd.DataFrame([claim_features]))[0, 1]
    return {
        "review_probability": round(float(probability), 3),
        "needs_review": bool(probability >= review_threshold),
    }
