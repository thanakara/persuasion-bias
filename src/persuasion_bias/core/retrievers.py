from typing import override

from pydantic import Field
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore


class PersuasivenessRetriever(BaseRetriever):
    """
    Custom retriever that can filter by persuasiveness and source.
    Similar to vectorstore.as_retriever() method using the MMR algorithm.
    """

    # BaseRetriever uses Pydantic
    vectorstore: VectorStore = Field(...)

    @override
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        source_filter: str | None = None,
        min_persuasiveness: float | None = None,
        k: int = 5,
    ) -> list[Document]:
        """Retrieves documents with optional filtering on source and persuasiveness"""

        # Overfetch - Filter - Return top-k
        search_k = min(k * 3, 50)
        docs = self.vectorstore.similarity_search(query, k=search_k)

        filtered_docs = []
        for doc in docs:
            if source_filter and doc.metadata.get("source") != source_filter:
                continue

            if min_persuasiveness is not None:
                if doc.metadata.get("persuasiveness_delta", 0) < min_persuasiveness:
                    continue

            filtered_docs.append(doc)

            # Break when we have enough documents
            if len(filtered_docs) >= k:
                break

        return filtered_docs[:k]
