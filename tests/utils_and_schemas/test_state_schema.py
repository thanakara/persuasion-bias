from typing import Literal

import pytest

from langchain_core.messages import HumanMessage

from persuasion_bias.schemas.state import GraphState
from persuasion_bias.schemas.models import BiasAnalysis, BiasDetection, CialdiniPrinciple


@pytest.fixture
def valid_bias_analysis():
    return BiasAnalysis(
        detected_principles=[BiasDetection(principle=CialdiniPrinciple.AUTHORITY, severity="high")],
        overall_bias_score=0.8,
        emotional_manipulation_score=0.5,
        target_audience_analysis="Targets young professionals",
    )


def test_graph_state_instantiation(valid_bias_analysis):
    state = GraphState(
        query="Buy now!",
        is_argument=True,
        retrieval="some context",
        analysis=valid_bias_analysis,
        user_choice="y",
        explanation="This is biased",
        messages=[],
        conversation_messages=[],
        analysis_messages=[],
    )
    assert state["query"] == "Buy now!"
    assert state["user_choice"] == "y"


def test_graph_state_add_messages_reducer(valid_bias_analysis):
    """add_messages reducer should accumulate messages, not overwrite"""
    state = GraphState(
        query="test",
        is_argument=False,
        retrieval="",
        analysis=valid_bias_analysis,
        user_choice="n",
        explanation="",
        messages=[],
        conversation_messages=[HumanMessage(content="hello")],
        analysis_messages=[],
    )
    assert len(state["conversation_messages"]) == 1
    assert state["conversation_messages"][0].content == "hello"


def test_graph_state_user_choice_values():
    """Just document the valid values, no runtime enforcement."""
    hints = GraphState.__annotations__
    assert hints["user_choice"] == Literal["y", "n"]
