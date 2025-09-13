import operator

from typing import Annotated, TypedDict

from langgraph.graph import MessagesState
from langchain_core.messages import BaseMessage

from persuasion_bias.utils.outputs import BiasAnalysis


# Baseline state for simple RAG
class BaselineState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]


# State for Bias analysis
class AnalysisState(MessagesState):
    query: str
    is_argument: bool
    retrieval: str
    analysis: BiasAnalysis


# Full Graph state [retrieval, analysis, explanation]
class GraphState(MessagesState):
    query: str
    is_argument: bool
    retrieval: str
    analysis: BiasAnalysis
    explanation: str
