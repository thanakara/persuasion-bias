import pandas as pd
from datasets import load_dataset
from langchain_core.document_loaders.base import BaseLoader
from langchain_core.documents import Document


class PersuasionDatasetLoader(BaseLoader):
    """Custom LangChain loader for the Anthropic Persuasion Dataset"""

    repo_id: str = "Anthropic/persuasion"

    def load_from_huggingface(self) -> list[Document]:
        """Load from HuggingFace and convert into LangChain Documents"""

        dataset = load_dataset(self.repo_id)
        frame = dataset["train"].to_pandas()
        frame = self._deduplicate_frame_by_argument(frame=frame)

        documents = []
        for row in frame.itertuples():
            persuasiveness_delta = row.rating_final - row.rating_initial
            metadata = {
                "source": row.source,
                "prompt_type": row.prompt_type,
                "rating_initial": row.rating_initial,
                "rating_final": row.rating_final,
                "persuasiveness_delta": persuasiveness_delta,
                "claim": row.claim,
                "argument": row.argument,
                "is_human": row.source == "Human",
                "is_persuasive": persuasiveness_delta > 0,
            }

            page_content = (
                f"Claim: {row.claim}\n"
                f"Argument: {row.argument}\n"
                f"Source: {row.source}\n"
                f"Initial Rating: {row.rating_initial}\n"
                f"Final Rating: {row.rating_final}\n"
                f"Persuasiveness Change: {row.rating_final - row.rating_initial}"
            )

            document = Document(page_content=page_content, metadata=metadata)
            documents.append(document)

        return documents

    @staticmethod
    def _deduplicate_frame_by_argument(frame: pd.DataFrame) -> pd.DataFrame:
        for column in ("rating_initial", "rating_final"):
            frame[column] = frame[column].apply(lambda e: eval(e[0]))

        frame.prompt_type = frame.prompt_type.fillna("Other")
        idx_unique = frame.argument.drop_duplicates().index
        return frame.iloc[idx_unique]
