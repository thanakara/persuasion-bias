from typing import NamedTuple
from logging import getLogger
from collections.abc import Iterator

import pandas as pd

from datasets import load_dataset
from langchain_core.documents import Document
from langchain_core.document_loaders.base import BaseLoader

logger = getLogger(__name__)


class PersuasionDatasetLoader(BaseLoader):
    """Custom LangChain loader for the Anthropic/persuasion HuggingFace dataset."""

    def __init__(self, repo_id: str = "Anthropic/persuasion") -> None:
        self.repo_id = repo_id

    def _pre_process(self, document: NamedTuple, idx: int) -> tuple[str, dict]:
        persuasiveness_delta = document.rating_final - document.rating_initial
        page_content = (
            f"Document ID: {idx}\n"
            f"Claim: {document.claim}\n"
            f"Argument: {document.argument}\n"
            f"Source: {document.source}\n"
            f"Persuasion type: {document.prompt_type}"
        )
        metadata = {
            "doc_id": idx,
            "source": document.source,
            "prompt_type": document.prompt_type,
            "rating_initial": document.rating_initial,
            "rating_final": document.rating_final,
            "persuasiveness_delta": persuasiveness_delta,
            "claim": document.claim,
            "argument": document.argument,
            "is_human": document.source == "Human",
            "is_persuasive": persuasiveness_delta > 0,
        }
        return page_content, metadata

    def lazy_load(self) -> Iterator[Document]:
        ds = load_dataset(self.repo_id, split="train")
        frame = ds.to_pandas()
        frame = self._deduplicate_frame_by_argument(frame)

        for row in frame.itertuples():
            idx = row.Index
            page_content, metadata = self._pre_process(row, idx)
            yield Document(page_content=page_content, metadata=metadata)

    @staticmethod
    def _deduplicate_frame_by_argument(frame: pd.DataFrame) -> pd.DataFrame:
        for column in ("rating_initial", "rating_final"):
            frame[column] = frame[column].apply(lambda e: int(e.split()[0]))

        frame.prompt_type = frame.prompt_type.fillna("Other")
        idx_unique = frame.argument.drop_duplicates().index
        return frame.iloc[idx_unique]

    @classmethod
    def load_documents(cls, repo_id: str = "Anthropic/persuasion") -> list[Document]:
        loader = cls(repo_id=repo_id)
        logger.info(f"Loading documents from {repo_id}...")
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} documents")

        return documents
