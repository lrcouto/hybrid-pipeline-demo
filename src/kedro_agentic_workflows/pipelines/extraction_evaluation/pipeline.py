from kedro.pipeline import Pipeline, llm_context_node, node, pipeline

from .nodes import init_field_evaluator, make_extraction_task, run_extraction_experiment


def create_pipeline(**kwargs) -> Pipeline:
    """Evaluate the LLM step that feeds the ML model. Run with --env langfuse."""
    return pipeline(
        [
            llm_context_node(
                name="extraction_context_node",
                outputs="extraction_context",
                llm="llm",
                prompts=["extraction_prompt"],
            ),
            node(make_extraction_task, "extraction_context", "extraction_task", name="make_extraction_task_node"),
            node(init_field_evaluator, None, "field_evaluator", name="init_field_evaluator_node"),
            node(
                func=run_extraction_experiment,
                inputs=["extraction_evaluation_data", "extraction_task", "field_evaluator", "params:model_name"],
                outputs="extraction_eval_summary",
                name="run_extraction_experiment_node",
            ),
        ]
    )
