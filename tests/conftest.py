import sys

from pathlib import Path

import pandas as pd
import pytest

from pytest_mock import MockerFixture

from persuasion_bias.retrieval.loader import PersuasionDatasetLoader

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def mock_embedding(mocker: MockerFixture):
    return mocker.MagicMock()


@pytest.fixture
def mock_loader(mocker: MockerFixture):
    loader = mocker.MagicMock()
    loader.load_documents.return_value = [mocker.MagicMock()]
    return loader


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
