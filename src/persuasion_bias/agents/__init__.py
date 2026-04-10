from .tools import get_time, get_tavily_tool, make_retrieve_tool


def make_tools(**kwargs):
    if not (retriever := kwargs.get("retriever")):
        raise ValueError("Need retriever to create tools")

    return [
        get_time,
        get_tavily_tool(),
        make_retrieve_tool(retriever),
    ]
