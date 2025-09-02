from datasets import load_dataset
from langchain_core.document_loaders.base import BaseLoader
from langchain_core.documents import Document


class PersuasionDatasetLoader(BaseLoader):
    """-----------------------------------------------------------+
    |Custom LangChain loader for the Anthropic Persuasion Dataset |
    +-----------------------------------------------------------"""

    repo_id: str = "Anthropic/persuasion"

    def load_from_huggingface(self) -> list[Document]:
        """Load from HuggingFace and convert into LangChain Documents"""

        dataset = load_dataset(self.repo_id)
        df = dataset["train"].to_pandas()
        df["rating_initial"] = df["rating_initial"].apply(lambda x: eval(x[0]))
        df["rating_final"] = df["rating_final"].apply(lambda x: eval(x[0]))
        #
        documents = []
        for idx, row in enumerate(df.itertuples()):
            metadata = {
                "id": idx,
                "source": row.source,
                "prompt_type": row.prompt_type,
                "rating_initial": row.rating_initial,
                "rating_final": row.rating_final,
                "persuasiveness_delta": row.rating_final - row.rating_initial,
                "claim": row.claim,
                "argument": row.argument,
                "is_human": row.source == "Human",
                "is_persuasive": ((row.rating_final - row.rating_initial) > 0),
            }

            page_content = (
                f"Claim: {row.claim}\n"
                f"Argument: {row.argument}\n"
                f"Source: {row.source}\n"
                f"Persuasiveness Change: {row.rating_final - row.rating_initial}\n"
                f"Initial Rating: {row.rating_initial}\n"
                f"Final Rating: {row.rating_final}"
            )

            document = Document(page_content=page_content, metadata=metadata)
            documents.append(document)

        return documents

    # def _deduplicate_documents_by_claim_argument(
    #     self,
    #     documents: list[Document],
    # ) -> list[Document]:
    #     seen = set()
    #     unique_docs = []

    #     for doc in documents:
    #         claim = doc.metadata.get("claim", "").strip()
    #         argument = doc.metadata.get("argument", "").strip()
    #         key = (claim, argument)

    #         if key not in seen:
    #             seen.add(key)
    #             unique_docs.append(doc)

    #     return unique_docs
