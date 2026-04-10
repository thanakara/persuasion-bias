from logging import getLogger
from operator import itemgetter as iget
from functools import cached_property

from langchain.tools import BaseTool
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage, HumanMessage
from langchain_core.runnables import Runnable
from langchain_core.language_models import BaseChatModel

logger = getLogger(__name__)


class ReActChain:
    def __init__(
        self,
        llm: BaseChatModel,
        prompts: dict[str, str],
        tools: list[BaseTool] | None = None,
        max_iterations: int = 2,  # 2 is often enough for a single loop
    ) -> None:
        self.llm = llm
        self.system_prompt = prompts.pop("system")
        self.tools = tools or []
        self.tools_dict = {t.name: t for t in self.tools}
        self._scratchpad: list[BaseMessage] = []
        self.max_iterations = max_iterations

    def __call__(self, input_: str) -> str:
        self._scratchpad.append(HumanMessage(input_))
        for _ in range(self.max_iterations):
            if answer := self._step():
                self._scratchpad.append(AIMessage(answer))
                return answer
        return "Max iterations reached."

    def bind(self, **kwargs):
        self.llm = self.llm.bind(**kwargs)

    @property
    def scratchpad(self) -> list[BaseMessage]:
        return self._scratchpad

    @property
    def chat_history(self) -> list[BaseMessage]:
        return [
            m for m in self._scratchpad
            if isinstance(m, HumanMessage)
            or (isinstance(m, AIMessage) and not m.tool_calls)
        ]  # fmt: skip

    @cached_property
    def chain(self) -> Runnable:
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(self.system_prompt),
                MessagesPlaceholder("agent_scratchpad"),
            ]
        )
        return (
            {
                "agent_scratchpad": iget("agent_scratchpad"),
            }
            | prompt
            | self.llm.bind_tools(self.tools)
        )

    def _observe(self, tc: dict) -> ToolMessage:
        tool_name = tc["name"]
        if tool_name not in self.tools_dict:
            content = f"Tool `{tool_name}` not found."
            logger.warning(content)
        else:
            try:
                args = tc["args"]
                content = self.tools_dict[tool_name].invoke(args)
            except Exception as e:
                content = f"Tool `{tool_name}` failed: {e}"
                logger.error(content)

        return ToolMessage(content=str(content), tool_call_id=tc["id"])

    def _step(self) -> str | bool:
        thought = self.chain.invoke({"agent_scratchpad": self._scratchpad})
        logger.info(f"Thought: tool_calls={[tc['name'] for tc in thought.tool_calls]}")

        if not thought.tool_calls:
            logger.info("No tool calls, returning direct answer")
            return thought.content

        observations = [self._observe(tc) for tc in thought.tool_calls]
        self._scratchpad += [thought] + observations

        logger.info(f"Tools called: {[tc['name'] for tc in thought.tool_calls]}")

        return False
