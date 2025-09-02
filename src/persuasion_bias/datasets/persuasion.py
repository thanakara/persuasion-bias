from datasets import load_dataset
from langchain_core.document_loaders.base import BaseLoader
from langchain_core.documents.base import Document


class PersuasionDatasetLoader(BaseLoader):
    """-----------------------------------------------------------+
    |Custom LangChain loader for the Anthropic Persuasion Dataset |
    +-----------------------------------------------------------"""

    repo_id: str = "Anthropic/persuasion"

    def load_and_preprocess(self) -> list[Document]:
        """Load from HuggingFace and convert into LangChain Documents"""

        dataset = load_dataset(self.repo_id)
        df = dataset["train"].to_pandas()
        df[["rating_initial", "rating_final"]] = df[
            ["rating_initial", "rating_final"]
        ].map(lambda e: eval(e[0]))
        documents = []
        for row in df.itertuples():
            metadata = {
                "source": row.source,
                "prompt_type": row.prompt_type,
                "rating_initial": row.rating_initial,
                "rating_final": row.rating_final,
                "persuasuveness_delta": row.rating_final - row.rating_initial,
                "claim": row.claim,
                "argument": row.argument,
                "is_human": row.source == "Human",
                "is_persuasive": ((row.rating_final - row.rating_initial) > 0),
            }

            page_content = f"""Claim: {row.claim}\nArgument: {row.argument}\nSource: {row.source}\
                \nPersuassiveness Change: {row.rating_final - row.rating_initial}\
                \nInitial Rating: {row.rating_initial}\nFinal Rating: {row.rating_final}"""

            document = Document(page_content=page_content, metadata=metadata)
            documents.append(document)

        return documents
