from functools import wraps
from typing import Callable

from IPython.display import Markdown, display
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph


def join_documents(documents: list[Document]) -> str:
    """Wraps over retrieval_tool function. Simply joins with newlines the documents."""

    result = []
    for document in documents:
        result.append(document.page_content)

    return "\n\n".join(result)


# Decorator over Callables that invoke the retriever and return List[Document]
def joins(func: Callable) -> Callable:
    @wraps("func")
    def _merge(*args, **kwargs) -> str:
        retrieval = func(*args, **kwargs)
        iterable = []

        for document in retrieval:
            iterable.append(document.page_content)
        return "\n---------\n".join(iterable)

    return _merge


def get_response_content(agent: CompiledStateGraph, argument: str) -> None:
    """Prettifier; display on the Markdown LLM response."""

    human_message = HumanMessage(argument)
    response = agent.invoke({"messages": [human_message]})

    display(Markdown(response.get("messages")[-1].content))
