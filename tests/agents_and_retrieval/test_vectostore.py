import pytest  # noqa: F401

from langchain_core.documents import Document

from persuasion_bias.retrieval.utils import join_documents
from persuasion_bias.retrieval.vectorstore import build_chroma, build_memory


# TODO: @pytest.mark.slow
def test_build_chroma_index_when_empty(mocker, mock_embedding, mock_loader):
    mock_vecstore = mocker.MagicMock()
    mock_vecstore.get.return_value = {"ids": {}}  # empty
    mocker.patch("persuasion_bias.retrieval.vectorstore.Chroma", return_value=mock_vecstore)

    build_chroma(mock_embedding, mock_loader, "test_collection", "/tmp/chroma")

    mock_loader.load_documents.assert_called_once()
    mock_vecstore.add_documents.assert_called_once()


def test_build_chroma_skips_indexing_when_populated(mocker, mock_embedding, mock_loader):
    mock_vecstore = mocker.MagicMock()
    mock_vecstore.get.return_value = {"ids": ["doc1", "doc2"]}  # populated
    mocker.patch("persuasion_bias.retrieval.vectorstore.Chroma", return_value=mock_vecstore)

    build_chroma(mock_embedding, mock_loader, "test_collection", "/tmp/chroma")

    mock_loader.load_documents.assert_not_called()
    mock_vecstore.add_documents.assert_not_called()


def test_build_chroma_returns_vectorstore(mocker, mock_embedding, mock_loader):
    mock_vectorstore = mocker.MagicMock()
    mock_vectorstore.get.return_value = {"ids": ["doc1"]}
    mocker.patch("persuasion_bias.retrieval.vectorstore.Chroma", return_value=mock_vectorstore)

    result = build_chroma(mock_embedding, mock_loader, "test_collection", "/tmp/chroma")

    assert result is mock_vectorstore


def test_build_memory_loads_documents(mocker, mock_embedding, mock_loader):
    mock_vs = mocker.MagicMock()
    mocker.patch("persuasion_bias.retrieval.vectorstore.InMemoryVectorStore.from_documents", return_value=mock_vs)

    result = build_memory(mock_embedding, mock_loader)

    mock_loader.load_documents.assert_called_once()
    assert result is mock_vs


def test_build_memory_passes_correct_args(mocker, mock_embedding, mock_loader):
    mock_from_docs = mocker.patch("persuasion_bias.retrieval.vectorstore.InMemoryVectorStore.from_documents")
    docs = mock_loader.load_documents.return_value

    build_memory(mock_embedding, mock_loader)

    mock_from_docs.assert_called_once_with(documents=docs, embedding=mock_embedding)


def test_join_documents():
    docs = [Document(page_content="foo"), Document(page_content="bar")]
    assert join_documents(docs) == "foo\n\nbar"
