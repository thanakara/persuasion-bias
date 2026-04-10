from uuid import uuid4

from langchain.tools import BaseTool
from langchain.agents import create_agent
from langgraph.graph.state import CompiledStateGraph
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.language_models import BaseChatModel


class ReActAPI:
    def __init__(
        self,
        llm: BaseChatModel,
        prompts: dict[str, str],
        tools: list[BaseTool] | None = None,
        memory: bool = True,
        thread_id: str | None = None,
    ) -> None:
        system_prompt = prompts.pop("system")

        checkpointer = InMemorySaver() if memory else None
        self.graph: CompiledStateGraph = create_agent(
            model=llm,
            tools=tools or [],
            system_prompt=SystemMessage(content=system_prompt),
            checkpointer=checkpointer,
        )
        self._default_thread = thread_id or str(uuid4())
        self._config = {"configurable": {"thread_id": self._default_thread}}

    def __call__(self, input_: str, *, thread_id: str | None = None) -> str:
        config = {**self._config, "configurable": {"thread_id": thread_id or self._default_thread}}
        result = self.graph.invoke({"messages": [HumanMessage(content=input_)]}, config=config)

        *_, last_msg = result["messages"]
        return last_msg.content

    @property
    def chat_history(self) -> list[BaseMessage]:
        state = self.graph.get_state(self._config)
        return state.values.get("messages", [])
