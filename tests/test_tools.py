from persuasion_bias.agents.tools import get_time, get_tavily_tool, make_retrieve_tool


def test_get_time_returns_string():
    assert isinstance(get_time.invoke({}), str)


def test_get_time_format(mocker):
    mock_now = mocker.patch("persuasion_bias.agents.tools.datetime")
    mock_now.now.return_value.strftime.return_value = "Monday 11-Apr-26 10:00AM"
    result = get_time.invoke({})
    assert result == "Monday 11-Apr-26 10:00AM"


def test_get_tavily_tool_returns_tool(mocker):
    mocker.patch("persuasion_bias.agents.tools.TavilySearch")
    tool = get_tavily_tool()
    assert tool is not None


def test_get_tavily_tool_max_results(mocker):
    mock_tavily = mocker.patch("persuasion_bias.agents.tools.TavilySearch")
    get_tavily_tool()
    mock_tavily.assert_called_once_with(max_results=2)


def test_make_retrieve_tool_returns_tool(mock_retriever):
    tool = make_retrieve_tool(mock_retriever)
    assert tool is not None
    assert tool.name == "retrieve"


def test_make_retrieve_tool_invokes_retriever(mock_retriever):
    tool = make_retrieve_tool(mock_retriever)
    tool.invoke({"query": "some-argument"})
    mock_retriever.invoke.assert_called_once_with("some-argument")


def test_make_retrieve_tool_joins_documents(mock_retriever):
    tool = make_retrieve_tool(mock_retriever)
    result = tool.invoke({"query": "test"})
    assert result == "doc1\n\ndoc2"
