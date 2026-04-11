from persuasion_bias.agents.edges import should_continue, should_make_retrieval


def test_should_continue_true_when_is_argument():
    assert should_continue({"is_argument": True}) == "true"


def test_should_continue_false_when_not_argument():
    assert should_continue({"is_argument": False}) == "false"


def test_should_make_retrieval_true_when_tool_calls(mocker):
    msg = mocker.MagicMock()
    msg.tool_calls = [mocker.MagicMock()]
    assert should_make_retrieval({"messages": [msg]}) is True


def test_should_make_retrieval_false_when_no_tool_calls(mocker):
    msg = mocker.MagicMock()
    msg.tool_calls = []
    assert should_make_retrieval({"messages": [msg]}) is False


def test_should_make_retrieval_uses_last_message(mocker):
    # spec=[] trick to test `hasattr` branch:
    first = mocker.MagicMock(spec=[])  # no tool-calls
    last = mocker.MagicMock()
    last.tool_calls = [mocker.MagicMock()]
    assert should_make_retrieval({"messages": [first, last]}) is True
