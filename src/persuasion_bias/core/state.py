import operator

# from langgraph.graph.message import add_messages
from typing import Annotated, List, TypedDict

from langchain_core.messages import BaseMessage


class BaselineState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
