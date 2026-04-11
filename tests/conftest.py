import pandas as pd
import pytest

from pytest_mock import MockerFixture
from langchain_core.documents import Document

from persuasion_bias.agents.react import ReActChain
from persuasion_bias.schemas.models import BiasAnalysis, BiasDetection, CialdiniPrinciple
from persuasion_bias.retrieval.loader import PersuasionDatasetLoader


@pytest.fixture
def loader():
    return PersuasionDatasetLoader()


@pytest.fixture
def sample_frame():
    return pd.DataFrame(
        {
            "argument": ["arg1", "arg2", "arg1"],  # arg1 is duplicated
            "rating_initial": ["3 - Somewhat oppose", "2 - Oppose", "4 - Neither oppose nor support"],
            "rating_final": ["5 - Somewhat support", "3 - Somewhat oppose", "6 - Support"],
            "prompt_type": ["Logical Reasoning", None, "Compelling Case"],
        }
    )


@pytest.fixture
def mock_llm(mocker: MockerFixture):
    return mocker.MagicMock()


@pytest.fixture
def prompts():
    return {
        "system": "You are a helpful assistant.",
        "other": "some-prompt",
    }


@pytest.fixture
def mock_retriever(mocker: MockerFixture):
    retriever = mocker.MagicMock()
    retriever.invoke.return_value = [
        Document(page_content="doc1"),
        Document(page_content="doc2"),
    ]
    return retriever


@pytest.fixture
def valid_bias_analysis():
    return BiasAnalysis(
        detected_principles=[BiasDetection(principle=CialdiniPrinciple.AUTHORITY, severity="high")],
        overall_bias_score=0.8,
        emotional_manipulation_score=0.5,
        target_audience_analysis="Targets young professionals",
    )


@pytest.fixture
def mock_tool(mocker: MockerFixture):
    tool = mocker.MagicMock()
    tool.name = "test_tool"
    tool.invoke.return_value = "tool-result"
    return tool


@pytest.fixture
def react_chain(mock_llm, prompts):
    return ReActChain(mock_llm, prompts)
