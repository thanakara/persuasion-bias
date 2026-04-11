from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from persuasion_bias.agents.nodes import (
    analyze_node,
    explain_node,
    classify_node,
    retrieval_node,
    route_classify,
    conversation_llm_node,
)


def test_classify_node_is_argument_true(mocker, mock_llm):
    mock_llm.with_structured_output.return_value.invoke.return_value = mocker.MagicMock(is_argument=True)
    state = {
        "messages": [HumanMessage("This is an argument")],
        "conversation_messages": [],
        "analysis_messages": [],
    }
    result = classify_node(state, mock_llm, "classify-prompt")

    assert result["is_argument"] is True
    assert result["query"] == "This is an argument"
    assert "analysis_messages" in result


def test_classify_node_is_argument_false(mocker, mock_llm):
    mock_llm.with_structured_output.return_value.invoke.return_value = mocker.MagicMock(is_argument=False)
    state = {
        "messages": [HumanMessage("Hello")],
        "conversation_messages": [],
        "analysis_messages": [],
    }

    result = classify_node(state, mock_llm, "classify prompt")

    assert result["is_argument"] is False
    assert "conversation_messages" in result


def test_classify_node_routes_message_to_correct_history(mocker, mock_llm):
    mock_llm.with_structured_output.return_value.invoke.return_value = mocker.MagicMock(is_argument=True)
    state = {
        "messages": [HumanMessage("arg")],
        "conversation_messages": [],
        "analysis_messages": [],
    }

    result = classify_node(state, mock_llm, "prompt")

    # is_argument=True -> analysis_messages, not conversation_messages
    assert "analysis_messages" in result
    assert "conversation_messages" not in result


def test_route_classify_true():
    assert route_classify({"is_argument": True}) == "true"


def test_route_classify_false():
    assert route_classify({"is_argument": False}) == "false"


def test_conversation_llm_node_returns_messages(mock_llm):
    response = AIMessage(content="response", tool_calls=[])
    mock_llm.invoke.return_value = response
    state = {"conversation_messages": [HumanMessage(content="hi")]}

    result = conversation_llm_node(state, mock_llm, "sys prompt")

    assert result["messages"] == [response]
    assert result["conversation_messages"] == [response]


def test_conversation_llm_node_prepends_system_message(mock_llm):
    mock_llm.invoke.return_value = AIMessage("ok", tool_calls=[])
    state = {"conversation_messages": [HumanMessage("hi")]}

    conversation_llm_node(state, mock_llm, "sys prompt")

    call_args = mock_llm.invoke.call_args[0][0]
    assert isinstance(call_args[0], SystemMessage)
    assert call_args[0].content == "sys prompt"


def test_retrieval_node_returns_messages(mock_llm):
    response = AIMessage(content="retrieved", tool_calls=[])
    mock_llm.invoke.return_value = response
    state = {"analysis_messages": [HumanMessage("arg")]}

    result = retrieval_node(state, mock_llm, "sys prompt")

    assert result["messages"] == [response]
    assert result["analysis_messages"] == [response]


def test_analyze_node_returns_analysis(mock_llm, valid_bias_analysis):
    mock_llm.with_structured_output.return_value.invoke.return_value = valid_bias_analysis
    state = {
        "query": "Buy now!",
        "analysis_messages": [AIMessage("retrieved context", tool_calls=[])],
    }

    result = analyze_node(state, mock_llm, "Query: {query}\nContext: {context}\n{format_instructions}")

    assert result["analysis"] is valid_bias_analysis


def test_analyze_node_uses_last_analysis_message(mock_llm, valid_bias_analysis):
    mock_llm.with_structured_output.return_value.invoke.return_value = valid_bias_analysis
    first = AIMessage("first", tool_calls=[])
    last = AIMessage("last context", tool_calls=[])
    state = {
        "query": "test",
        "analysis_messages": [first, last],
    }

    analyze_node(state, mock_llm, "Query: {query}\nContext: {context}\n{format_instructions}")

    prompt_used = mock_llm.with_structured_output.return_value.invoke.call_args[0][0]
    assert "last context" in prompt_used


def test_explain_node_returns_analysis_json_when_user_declines(mocker, valid_bias_analysis):
    state = {
        "user_choice": "n",
        "analysis": valid_bias_analysis,
        "query": "test",
    }
    result = explain_node(state, mocker.MagicMock(), "prompt")

    assert result["messages"][0].content == valid_bias_analysis.model_dump_json(indent=2)


def test_explain_node_invokes_model_when_user_accepts(mock_llm, valid_bias_analysis):
    ai_msg = AIMessage("Here is the explanation")
    mock_llm.invoke.return_value = ai_msg
    state = {
        "user_choice": "y",
        "analysis": valid_bias_analysis,
        "query": "Buy now!",
    }

    result = explain_node(state, mock_llm, "Query: {query}\nAnalysis: {analysis}")

    assert result["explanation"] == "Here is the explanation"
    assert result["messages"] == [ai_msg]
