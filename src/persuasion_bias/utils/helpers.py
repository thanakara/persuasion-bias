from IPython.display import Markdown, display
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph


def documents_reader_helper(documents: list[Document]) -> str:
    """Wraps over retrieval_tool function. It simply joins with newlines the documents."""

    result = []
    for document in documents:
        result.append(document.page_content)

    return "\n\n".join(result)


def get_llama_response_content(agent: CompiledStateGraph, argument: str) -> None:
    """Prettifier display on the markdown LLM response."""

    human_message = HumanMessage(argument)
    response = agent.invoke({"messages": [human_message]})

    display(Markdown(response.get("messages")[-1].content))
