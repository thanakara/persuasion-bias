from logging import getLogger

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore, InMemoryVectorStore
from langchain_core.document_loaders import BaseLoader

logger = getLogger(__name__)


def build_chroma(
    embedding: Embeddings,
    loader: BaseLoader,
    collection_name: str,
    persist_directory: str,
) -> VectorStore:
    """Build or load a persistent Chroma vectorstore."""
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embedding,
        persist_directory=persist_directory,
    )
    if not vectorstore.get()["ids"]:
        logger.info("Empty collection, indexing documents...")
        vectorstore.add_documents(loader.load_documents())
    else:
        logger.info("Collection already has documents, skipping indexing")
    return vectorstore


def build_memory(embedding: Embeddings, loader: BaseLoader) -> VectorStore:
    """Build an ephemeral in-memory vectorstore."""
    logger.info("Using in-memory vectorstore")
    return InMemoryVectorStore.from_documents(documents=loader.load_documents(), embedding=embedding)
