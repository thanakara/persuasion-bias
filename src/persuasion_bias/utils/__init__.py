from enum import StrEnum

from langchain_core.messages import BaseMessage

from persuasion_bias.schemas.state import GraphState

from .path import CONFIG_DIR
from .device import decide_device


def get_last_message(state: GraphState, source: str) -> BaseMessage:
    return state[source][-1]


class Turn(StrEnum):
    USER = "👤"
    ASSISTANT = "🤖"


__all__ = ["CONFIG_DIR", "decide_device", "Turn", "get_last_message"]
