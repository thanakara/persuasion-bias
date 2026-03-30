from .tools import get_time, multiply_numbers, make_retrieve_tool


def make_tools(**kwargs):
    if not (retriever := kwargs.get("retriever")):
        raise ValueError("Need retriever to create tools")

    return [
        multiply_numbers,
        get_time,
        make_retrieve_tool(retriever),
    ]
