from logging import getLogger
from datetime import datetime

from pydantic import BaseModel
from langchain.tools import BaseTool, tool
from langchain_tavily import TavilySearch

from persuasion_bias.retrieval.utils import join_documents

logger = getLogger(__name__)


class MultiplyInput(BaseModel):
    a: float
    b: float


@tool(
    "get_time",
    description=(
        "Use this tool when the user asks about the current time or date. "
        "Returns the current local date and time. "
        "Do not answer time-related questions from memory, always use this tool."
    ),
)
def get_time() -> str:
    return datetime.now().strftime("%A %d-%b-%y %I:%M%p")


def get_tavily_tool() -> BaseTool:
    return TavilySearch(max_results=2)


def make_retrieve_tool(retriever) -> BaseTool:
    @tool(
        "retrieve",
        description=(
            "Retrieve persuasion techniques, logical fallacies, and bias examples from the knowledge base. "
            "Call this tool EXACTLY ONCE per turn using the user's original argument as the query. "
            "Always reference the document IDs in your reasoning and final answer, and use them to support your claims."
        ),
    )
    def retrieve(query: str) -> str:
        logger.debug(f"Retrieving documents for query: {query}")
        docs = retriever.invoke(query)
        return join_documents(docs)

    return retrieve
