from typing import Literal

from persuasion_bias.schemas.state import GraphState, BaselineState


def should_continue(state: GraphState) -> Literal["true", "false"]:
    return ("false", "true")[state.get("is_argument")]


def should_make_retrieval(state: BaselineState) -> bool:
    """Returns True if the last LLM message contains tool calls."""
    *_, last_message = state.get("messages")
    return hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0
