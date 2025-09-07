import operator

from langgraph.graph.message import add_messages
from typing import Annotated, TypedDict, Sequence

from langchain_core.messages import BaseMessage, AnyMessage
from persuasion_bias.utils.outputs import BiasAnalysis



# Baseline state for simple RAG
class BaselineState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]


# Analysis state
class BiasAnalysisState(TypedDict):
    messages: Annotated[Sequence[AnyMessage], add_messages]
    # analysis: BiasAnalysis
