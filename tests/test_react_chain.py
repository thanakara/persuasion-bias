from langchain_core.messages import AIMessage, ToolMessage, HumanMessage

from persuasion_bias.agents.react import ReActChain


def test_init_pops_system_prompt(mock_llm):
    prompts = {"system": "sys", "other": "other"}
    ReActChain(mock_llm, prompts)
    assert "system" not in prompts


def test_init_empty_tools(react_chain):
    assert react_chain.tools == []
    assert react_chain.tools_dict == {}


def test_init_tools_dict(mock_llm, prompts, mock_tool):
    chain = ReActChain(llm=mock_llm, prompts=prompts, tools=[mock_tool])
    assert "test_tool" in chain.tools_dict


def test_init_default_max_iterations(react_chain):
    assert react_chain.max_iterations == 2  # noqa: PLR2004


def test_call_returns_answer_on_first_step(mocker, react_chain):
    react_chain._step = mocker.MagicMock(return_value="direct answer")
    result = react_chain("hello")
    assert result == "direct answer"


def test_call_appends_human_message(mocker, react_chain):
    react_chain._step = mocker.MagicMock(return_value="answer")
    react_chain("hello")
    assert isinstance(react_chain._scratchpad[0], HumanMessage)
    assert react_chain._scratchpad[0].content == "hello"


def test_call_appends_ai_message_on_answer(mocker, react_chain):
    react_chain._step = mocker.MagicMock(return_value="answer")
    react_chain("hello")
    assert any(isinstance(m, AIMessage) and m.content == "answer" for m in react_chain._scratchpad)


def test_call_returns_max_iterations_message(mocker, react_chain):
    react_chain._step = mocker.MagicMock(return_value=False)
    result = react_chain("hello")
    assert result == "Max iterations reached."


def test_call_respects_max_iterations(mocker, react_chain):
    react_chain._step = mocker.MagicMock(return_value=False)
    react_chain.max_iterations = 3
    react_chain("hello")
    assert react_chain._step.call_count == 3  # noqa: PLR2004


def test_chat_history_includes_human_messages(react_chain):
    react_chain._scratchpad = [HumanMessage(content="hi")]
    assert len(react_chain.chat_history) == 1


def test_chat_history_includes_ai_messages_without_tool_calls(react_chain):
    react_chain._scratchpad = [AIMessage(content="answer", tool_calls=[])]
    assert len(react_chain.chat_history) == 1


def test_chat_history_excludes_ai_messages_with_tool_calls(mocker, react_chain):
    ai_msg = mocker.MagicMock(spec=AIMessage)
    ai_msg.tool_calls = [{"name": "tool"}]
    react_chain._scratchpad = [ai_msg]
    assert len(react_chain.chat_history) == 0


def test_observe_calls_tool(react_chain, mock_tool):
    react_chain.tools_dict = {"test_tool": mock_tool}
    tc = {"name": "test_tool", "args": {"query": "test"}, "id": "123"}

    result = react_chain._observe(tc)

    mock_tool.invoke.assert_called_once_with({"query": "test"})
    assert isinstance(result, ToolMessage)
    assert result.content == "tool-result"


def test_observe_returns_error_message_for_unknown_tool(react_chain):
    tc = {"name": "unknown_tool", "args": {}, "id": "123"}
    result = react_chain._observe(tc)
    assert "not found" in result.content


def test_observe_handles_tool_exception(react_chain, mock_tool):
    mock_tool.invoke.side_effect = Exception("tool error")
    react_chain.tools_dict = {"test_tool": mock_tool}
    tc = {"name": "test_tool", "args": {}, "id": "123"}

    result = react_chain._observe(tc)
    assert "failed" in result.content


def test_step_returns_content_when_no_tool_calls(mocker, react_chain):
    react_chain.chain = mocker.MagicMock()
    react_chain.chain.invoke.return_value = AIMessage(content="direct", tool_calls=[])

    result = react_chain._step()
    assert result == "direct"


def test_step_returns_false_when_tool_calls(mocker, react_chain, mock_tool):
    react_chain.tools_dict = {"test_tool": mock_tool}
    react_chain.chain = mocker.MagicMock()

    ai_msg = AIMessage(content="", tool_calls=[{"name": "test_tool", "args": {}, "id": "1"}])
    react_chain.chain.invoke.return_value = ai_msg

    result = react_chain._step()
    assert result is False


def test_step_appends_thought_and_observations_to_scratchpad(mocker, react_chain, mock_tool):
    react_chain.tools_dict = {"test_tool": mock_tool}
    react_chain.chain = mocker.MagicMock()

    ai_msg = AIMessage(content="", tool_calls=[{"name": "test_tool", "args": {}, "id": "1"}])
    react_chain.chain.invoke.return_value = ai_msg

    react_chain._step()

    assert ai_msg in react_chain._scratchpad
    assert any(isinstance(m, ToolMessage) for m in react_chain._scratchpad)
