from kedro.pipeline import Pipeline, llm_context_node, node, pipeline

from .nodes import extract_claim_features, predict_review_risk


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            # The LLM and the prompt come from the catalog, like any other data.
            llm_context_node(
                name="extraction_context_node",
                outputs="extraction_context",
                llm="llm",
                prompts=["extraction_prompt"],
            ),
            node(
                func=extract_claim_features,
                inputs=["extraction_context", "params:message", "intent_detection_result"],
                outputs="claim_features",
                name="extract_claim_features_node",
            ),
            node(
                func=predict_review_risk,
                inputs=["triage_model", "claim_features", "params:triage.review_threshold"],
                outputs="triage_prediction",
                name="predict_review_risk_node",
            ),
        ]
    )
