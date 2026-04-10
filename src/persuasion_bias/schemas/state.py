from typing import Literal, Annotated

from langgraph.graph import MessagesState, add_messages
from langchain_core.messages import BaseMessage

from persuasion_bias.schemas.models import BiasAnalysis


class GraphState(MessagesState):
    query: str
    is_argument: bool
    retrieval: str
    analysis: BiasAnalysis
    user_choice: Literal["y", "n"]
    explanation: str
    # branch-specific message histories
    conversation_messages: Annotated[list[BaseMessage], add_messages]
    analysis_messages: Annotated[list[BaseMessage], add_messages]
