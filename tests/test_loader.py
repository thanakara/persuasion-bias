# ruff: noqa : PLR2004
from collections import namedtuple

import pandas as pd
import pytest  # noqa: F401

from langchain_core.documents import Document

from persuasion_bias.retrieval.loader import PersuasionDatasetLoader

FakeRow = namedtuple("FakeRow", ["claim", "argument", "source", "prompt_type", "rating_initial", "rating_final"])


# TODO: @pytest.mark.slow
def test_pre_process_page_content(loader):
    row = FakeRow("Climate change is real", "Evidence shows...", "Human", "logical", 3, 5)
    page_content, _ = loader._pre_process(row, idx=0)

    assert "Climate change is real" in page_content
    assert "Evidence shows..." in page_content
    assert "Human" in page_content
    assert "logical" in page_content
    assert "Document ID: 0" in page_content


def test_pre_process_metadata_fields(loader):
    row = FakeRow("Claim", "Argument", "Human", "emotional", 2, 6)
    _, metadata = loader._pre_process(row, idx=1)

    assert metadata["persuasiveness_delta"] == 4
    assert metadata["is_human"] is True
    assert metadata["is_persuasive"] is True
    assert metadata["rating_initial"] == 2
    assert metadata["rating_final"] == 6


def test_pre_process_not_persuasive(loader):
    row = FakeRow("Claim", "Argument", "AI", "logical", 5, 3)
    _, metadata = loader._pre_process(row, idx=2)

    assert metadata["persuasiveness_delta"] == -2
    assert metadata["is_persuasive"] is False
    assert metadata["is_human"] is False


def test_pre_process_delta_zero_not_persuasive(loader):
    row = FakeRow("Claim", "Argument", "Human", "logical", 4, 4)
    _, metadata = loader._pre_process(row, idx=3)

    assert metadata["persuasiveness_delta"] == 0
    assert metadata["is_persuasive"] is False


def test_deduplication_removes_duplicates(sample_frame):
    result = PersuasionDatasetLoader._deduplicate_frame_by_argument(sample_frame)
    assert result["argument"].duplicated().sum() == 0


def test_deduplication_parses_ratings(sample_frame):
    result = PersuasionDatasetLoader._deduplicate_frame_by_argument(sample_frame)
    assert result["rating_initial"].dtype in (int, "int64")
    assert result["rating_final"].dtype in (int, "int64")


def test_deduplication_fills_prompt_type(sample_frame):
    result = PersuasionDatasetLoader._deduplicate_frame_by_argument(sample_frame)
    assert result["prompt_type"].isna().sum() == 0
    assert "Other" in result["prompt_type"].values


def test_lazy_load_yields_documents(mocker, loader):
    mock_ds = mocker.MagicMock()
    mock_ds.to_pandas.return_value = pd.DataFrame(
        {
            "argument": ["arg1"],
            "rating_initial": ["3 - Somewhat oppose"],
            "rating_final": ["5 - Somewhat support"],
            "prompt_type": ["Logical Reasoning"],
            "claim": ["claim1"],
            "source": ["Human"],
        }
    )
    mocker.patch("persuasion_bias.retrieval.loader.load_dataset", return_value=mock_ds)

    docs = list(loader.lazy_load())
    assert len(docs) == 1
    assert isinstance(docs[0], Document)


def test_load_documents_returns_list(mocker):
    mock_ds = mocker.MagicMock()
    mock_ds.to_pandas.return_value = pd.DataFrame(
        {
            "argument": ["arg1"],
            "rating_initial": ["3 - Somewhat oppose"],
            "rating_final": ["5 - Somewhat support"],
            "prompt_type": ["Logical Reasoning"],
            "claim": ["claim1"],
            "source": ["Human"],
        }
    )
    mocker.patch("persuasion_bias.retrieval.loader.load_dataset", return_value=mock_ds)

    docs = PersuasionDatasetLoader.load_documents()
    assert isinstance(docs, list)
    assert all(isinstance(d, Document) for d in docs)
