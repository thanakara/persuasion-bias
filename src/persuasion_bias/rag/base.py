from typing import Callable

from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.runnables import RunnableBinding
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from persuasion_bias.core.state import BaselineState
from persuasion_bias.utils.prompts import RAG_SYSTEM_MESSAGE


class BaselinePersuasionRAG:
    def __init__(self, llm: RunnableBinding, tools_dict: dict[str, Callable]) -> None:
        self.llm = llm
        self.tools_dict = tools_dict

    def _graph_fabricate(self) -> CompiledStateGraph:
        """Builds the LangGraph workflow."""

        flow = StateGraph(BaselineState)
        flow.add_node("LLM", self.call_llm)
        flow.add_node("RetrievalAgent", self.retriever_node)

        flow.add_conditional_edges(
            source="LLM",
            path=self.should_make_retrieval,
            path_map={True: "RetrievalAgent", False: END},
        )
        flow.add_edge("RetrievalAgent", "LLM")
        flow.set_entry_point("LLM")

        return flow.compile()

    @staticmethod
    def should_make_retrieval(state: BaselineState) -> bool:
        """Decides whether to retrieve documents or END. Returns boolean."""

        *_, last_message = state.get("messages")

        return hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0

    def call_llm(self, state: BaselineState) -> BaselineState:
        """Invokes the LLM to get a response, adding it to the state."""

        messages = state.get("messages")
        content = RAG_SYSTEM_MESSAGE
        system_message = SystemMessage(content=content)
        messages_add = [system_message] + messages
        response = self.llm.invoke(messages_add)

        return {"messages": [response]}

    def retriever_node(self, state: BaselineState) -> BaselineState:
        """Node that executes the retriever tool."""

        *_, last_message = state.get("messages")
        tool_calls = last_message.tool_calls

        results = []
        for t in tool_calls:
            if t.get("name") not in self.tools_dict:
                result = (
                    "Incorrect Tool Name. Please Retry and Select from Available Tools."
                )

            # Llama3.2 models after invocation sometimes return nested dictionaries
            try:
                assert "properties" in t.get("args")
                result = self.tools_dict.get(t.get("name"))(
                    **t.get("args").get("properties")
                )
            except AssertionError:
                result = self.tools_dict.get(t.get("name"))(**t.get("args"))

            tool_message = ToolMessage(content=str(result), tool_call_id=t.get("id"))
            results.append(tool_message)

        return {"messages": results}
