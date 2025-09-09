from typing import List

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import (
    DocumentCompressorPipeline,
    EmbeddingsFilter,
)
from langchain.vectorstores import VectorStore
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableSequence
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEmbeddings,
    HuggingFaceEndpoint,
)
from transformers import AutoTokenizer

from persuasion_bias.core.document_loaders import PersuasionDatasetLoader
from persuasion_bias.utils.prompts import RAG_PROMPT


class ChainRAGHuggingFace:
    """
    +------------------------------------------------+
    | Simple one-turn RAG runnable using open source |
    +------------------------------------------------+
    """

    def __init__(
        self,
        model_repo_id: str,
        embedding_model_repo_id: str,
        vector_storage: VectorStore = Chroma,
    ) -> None:
        self.model_repo_id = model_repo_id
        self.embedding_model_repo_id = embedding_model_repo_id
        self.vector_storage = vector_storage

    def _load_models_from_hub(
        self,
    ) -> tuple[HuggingFaceEmbeddings, ChatHuggingFace, AutoTokenizer]:
        """Loads Models and Tokenizer from HuggingFace"""

        embedding = HuggingFaceEmbeddings(
            model=self.embedding_model_repo_id,
            model_kwargs={"device": "cpu"},  # TODO: FAISS: -supports GPU
            encode_kwargs={"normalize_embeddings": True},
        )
        endpoint = HuggingFaceEndpoint(
            repo_id=self.model_repo_id,
            task="text-generation",
            max_new_tokens=512,
            do_sample=False,
            repetition_penalty=1.03,
        )
        chat_model = ChatHuggingFace(llm=endpoint)

        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.model_repo_id, trust_remote_code=True
        )

        return (embedding, chat_model, tokenizer)

    def _build_prompt(self) -> ChatPromptTemplate:
        """Crete prompts using the model's Tokenizer"""

        system_prompt = RAG_PROMPT

        *_, tokenizer = self._load_models_from_hub()
        template = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "{question}"},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

        return ChatPromptTemplate.from_template(template=template)

    @staticmethod
    def _load_documents_from_hub() -> List[Document]:
        """Getting the Anthropic/persuasion dataset using custom class."""

        loader = PersuasionDatasetLoader()

        return loader.load_from_huggingface()

    def _vectorstore_as_retriever(self) -> VectorStoreRetriever:
        """Make the Retriever using preferenced vector store."""

        documents = self._load_documents_from_hub()
        embedding, *_ = self._load_models_from_hub()
        vectorstore = self.vector_storage.from_documents(
            documents=documents, embedding=embedding
        )

        return vectorstore.as_retriever(search_kwargs={"k": 3})

    def create_runnable_sequence(self) -> RunnableSequence:
        """RAG Chain. Simple one turn --no-memory."""

        _, llm, _ = self._load_models_from_hub()
        prompt = self._build_prompt()
        retriever = self._vectorstore_as_retriever()

        runnable = (
            {"question": RunnablePassthrough(), "documents": retriever}
            | prompt
            | llm
            | StrOutputParser()
        )

        return runnable


class CompressionRAGChain:
    """
    +---------------------------------------------------------------+
    | One-turn RAG runnable with contextual compression retrieval.  |
    | Optionally selects:                                           |
    |       - Claude 3.5 Haiku                                      |
    |       - similarity_threshold for relevance filtering          |
    +---------------------------------------------------------------+
    """

    def __init__(
        self,
        repo_id: str | None = None,
        vectorstore: VectorStore = Chroma,
        use_anthropic: bool = False,
        documents: List[Document] = [],
        retriever: VectorStoreRetriever = None,
    ) -> None:
        if repo_id and use_anthropic:
            raise ValueError("You cannot set both `repo_id` and `use_anthropic=True`.")

        if repo_id:
            use_anthropic = False
            llm = HuggingFaceEndpoint(repo_id=repo_id, temperature=0.1)
            self.model = ChatHuggingFace(llm=llm)

        elif use_anthropic:
            model_name = "claude-3-5-haiku-20241022"
            self.model = ChatAnthropic(temperature=0.1, model_name=model_name)

        else:
            raise ValueError(
                "You must specify either `repo_id` or set `use_anthropic`."
            )

        self.vectorstore = vectorstore
        self._documents = documents
        self._retriever = retriever

    def _load_documents_from_hub(self) -> List[Document]:
        """Loads Anthropic/persuasion using custom class."""

        if not self._documents:
            loader = PersuasionDatasetLoader()
            documents = loader.load_from_huggingface()
            self._documents.extend(documents)

        return self._documents

    @staticmethod
    def _load_sentence_transformers_embeddings() -> HuggingFaceEmbeddings:
        """Loads specific embedding model from HuggingFace."""

        kwargs = {
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "model_kwargs": {"device": "cpu"},
            "encode_kwargs": {"normalize_embeddings": False},
        }
        embeddings = HuggingFaceEmbeddings(**kwargs)
        return embeddings

    def _get_contextual_compression_retriever(
        self, similarity_threshold: float = 0.55
    ) -> ContextualCompressionRetriever:
        """Creates a contextual compression retriever and caches it."""

        if self._retriever is None:
            embedding = self._load_sentence_transformers_embeddings()
            documents = self._load_documents_from_hub()
            vectorstore = self.vectorstore.from_documents(
                documents=documents, embedding=embedding
            )
            vectorstore_retriever = vectorstore.as_retriever(
                search_type="mmr", search_kwargs={"k": 5}
            )

            redundant_filter = EmbeddingsRedundantFilter(embeddings=embedding)
            relevance_filter = EmbeddingsFilter(
                embeddings=embedding, similarity_threshold=similarity_threshold
            )
            compression_pipeline = DocumentCompressorPipeline(
                transformers=[redundant_filter, relevance_filter]
            )
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compression_pipeline,
                base_retriever=vectorstore_retriever,
            )
            self._retriever = compression_retriever

        return self._retriever

    @staticmethod
    def _create_prompt() -> ChatPromptTemplate:
        """Dynamic one-turn prompt."""

        system_prompt = RAG_PROMPT

        return ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("human", "{argument}")]
        )

    def create_runnable_sequence(self) -> RunnableSequence:
        """The full ContextualCompression RAG Chain --no-memory."""

        prompt = self._create_prompt()
        retriever = self._get_contextual_compression_retriever()
        model = self.model
        runnable = (
            {"argument": RunnablePassthrough(), "documents": retriever}
            | prompt
            | model
            | StrOutputParser()
        )

        return runnable
