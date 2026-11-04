from kedro.pipeline import Pipeline, node, pipeline

from .nodes import evaluate_model, split_data, train_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_data,
                inputs=["claim_history", "params:triage"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data_node",
            ),
            node(
                func=train_model,
                inputs=["X_train", "y_train", "params:triage"],
                outputs="triage_model",
                name="train_model_node",
            ),
            node(
                func=evaluate_model,
                inputs=["triage_model", "X_test", "y_test"],
                outputs="triage_metrics",
                name="evaluate_model_node",
            ),
        ]
    )
