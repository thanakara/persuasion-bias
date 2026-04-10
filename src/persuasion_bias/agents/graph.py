from uuid import uuid4
from typing import Literal
from functools import partial

from langchain.tools import BaseTool
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.state import CompiledStateGraph
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.language_models import BaseChatModel

from persuasion_bias.utils import get_last_message
from persuasion_bias.agents.nodes import (
    analyze_node,
    explain_node,
    classify_node,
    retrieval_node,
    route_classify,
    conversation_llm_node,
)
from persuasion_bias.schemas.state import GraphState


class BiasExplanationGraph:
    def __init__(
        self,
        llm: BaseChatModel,
        prompts: dict[str, str],
        tools: list[BaseTool] | None = None,
        thread_id: str | None = None,
    ):
        assert tools
        self._default_thread = thread_id or str(uuid4())
        self._config = {"configurable": {"thread_id": self._default_thread}}

        self.conversational_tools = [tool for tool in tools if tool.name != "retrieve"]
        self.retrieval_tool = [tool for tool in tools if tool.name == "retrieve"]

        self.prompts = prompts

        self.llm = llm
        self.conversation_model = llm.bind_tools(self.conversational_tools)
        self.retrieval_model = llm.bind_tools(self.retrieval_tool)

        self.graph = self._fabricate_graph()

    def __call__(self, input_: str, *, thread_id: str | None = None) -> str:
        config = {**self._config, "configurable": {"thread_id": thread_id or self._default_thread}}
        state = self.graph.invoke({"messages": [HumanMessage(content=input_)]}, config=config)

        snapshot = self.graph.get_state(config=config)
        if "explain" in snapshot.next:
            user_choice = self.get_user_choice()
            self.graph.update_state(config, {"user_choice": user_choice})
            state = self.graph.invoke(None, config=config)

            return get_last_message(state, "analysis_messages").content

        return get_last_message(state, "conversation_messages").content

    def _fabricate_graph(self) -> CompiledStateGraph:
        workflow = StateGraph(GraphState)

        self._add_nodes(workflow)
        self._add_edges(workflow)

        return workflow.compile(checkpointer=InMemorySaver(), interrupt_after=["analyze"])

    def _add_nodes(self, workflow: StateGraph):
        workflow.add_node(
            "classify",
            partial(
                classify_node,
                model=self.llm,
                prompt=self.prompts["is_argument"],
            ),
        )
        workflow.add_node(
            "conversation_llm",
            partial(
                conversation_llm_node,
                model=self.conversation_model,
                prompt=self.prompts["conversation"],
            ),
        )
        workflow.add_node(
            "retrieval_llm",
            partial(
                retrieval_node,
                model=self.retrieval_model,
                prompt=self.prompts["retrieval"],
            ),
        )
        workflow.add_node("retrieval_tool", ToolNode(tools=self.retrieval_tool))
        workflow.add_node(
            "analyze",
            partial(
                analyze_node,
                model=self.llm,
                prompt=self.prompts["analysis"],
            ),
        )
        workflow.add_node(
            "explain",
            partial(
                explain_node,
                model=self.llm,
                prompt=self.prompts["explanatory"],
            ),
        )
        workflow.add_node("tools", ToolNode(tools=self.conversational_tools))

    def _add_edges(self, workflow: StateGraph):
        workflow.add_edge("tools", "conversation_llm")
        workflow.add_conditional_edges(
            "classify",
            path=route_classify,
            path_map={
                "true": "retrieval_llm",
                "false": "conversation_llm",
            },
        )

        workflow.add_edge("retrieval_llm", "retrieval_tool")
        workflow.add_edge("retrieval_tool", "analyze")
        workflow.add_edge("analyze", "explain")
        workflow.add_edge("explain", END)

        workflow.set_entry_point("classify")
        workflow.add_conditional_edges("conversation_llm", tools_condition)

    @staticmethod
    def get_user_choice() -> Literal["y", "n"]:
        while True:
            input_ = input("Verbose (y) or JSON output (n)? [y/n]: ").strip().lower()
            if input_ in {"y", "n"}:
                return input_
            print(f"Invalid input '{input_}' — please enter 'y' or 'n'.")
