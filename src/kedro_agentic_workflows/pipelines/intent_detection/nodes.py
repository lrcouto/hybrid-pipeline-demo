import logging
from typing import Any

from kedro.pipeline import LLMContext
from kedro.pipeline.preview_contract import MermaidPreview
from langchain_core.messages import HumanMessage
import pandas as pd
from sqlalchemy import Engine

from .agent import IntentDetectionAgent
from ...utils import log_message

logger = logging.getLogger(__name__)


def generate_mermaid_preview() -> MermaidPreview:
    compiled = IntentDetectionAgent.graph().compile()
    mermaid = compiled.get_graph().draw_mermaid()
    # Override the 'first' class to have a visible background
    mermaid = mermaid.replace(
        "classDef first fill-opacity:0",
        "classDef first fill:#50C878,color:#000000",
    )
    config = {
        "themeVariables": {
            "lineColor": "#F5A623",
            "nodeTextColor": "#000000",
        },
    }
    return MermaidPreview(content=mermaid, meta=config)


def create_session(user_id: int) -> pd.DataFrame:
    """
    Create a new user session entry.

    Args:
        user_id: ID of the active user.

    Returns:
        DataFrame containing the new session entry.
    """
    return pd.DataFrame({"user_id": [user_id]})


def get_session_id(session_table: pd.DataFrame) -> int:
    """
    Extract the latest session ID from the session table.

    Args:
        session_table: DataFrame of sessions.

    Returns:
        Latest session ID.
    """
    session_id = int(session_table.iloc[-1]["id"])
    started_at = pd.to_datetime(session_table.iloc[-1]["started_at"])
    logger.info(
        "Session session_id %s created at started_at %s", session_id, started_at
    )
    return session_id


def load_context(
    user_id: int,
    user_data: pd.DataFrame,
    session_id: int,
    intent_tracer: Any,
):
    """
    Build user context and tracing configuration for LangChain.

    Args:
        user_id: ID of the active user.
        user_data: DataFrame containing user profile info.
        session_id: Current session ID.
        intent_tracer: LangChain callback from the active env's TraceDataset,
            or None when tracing is off (base env).

    Returns:
        Tuple of (user_context, session_config).
    """
    user_context = {"profile": {"user_id": user_id, "name": user_data.at[0, "name"]}}
    session_config = {
        "configurable": {"thread_id": str(session_id)},
        "callbacks": [intent_tracer] if intent_tracer else [],
    }
    return user_context, session_config


def detect_intent(
    intent_detection_context: LLMContext,
    message: str,
    user_context: dict,
    session_config: dict,
) -> dict[str, Any]:
    """
    Classify the customer's message into an intent.

    Args:
        intent_detection_context: LLM + intent prompt, built by the LLM context node.
        message: The customer's free-text message (a runtime parameter).
        user_context: Dictionary of user profile/context data.
        session_config: LLM session configuration (thread + callbacks).

    Returns:
        Intent detection result dictionary.
    """
    agent = IntentDetectionAgent(context=intent_detection_context)
    agent.compile()

    context = {"messages": [HumanMessage(content=message)], "user_context": user_context}
    result = agent.invoke(context, session_config)
    logger.info("Intent: %s (%s)", result["intent"], result["reason"])
    return result


def log_intent_detection(
    db_engine: Engine, session_id: int, intent_detection_result: dict
):
    """
    Persist intent detection conversation messages to the database.

    Args:
        db_engine: Active SQLAlchemy engine.
        session_id: Current session ID.
        intent_detection_result: Result dict from intent detection agent.
    """
    for m in intent_detection_result["messages"]:
        log_message(db_engine, session_id, m)
