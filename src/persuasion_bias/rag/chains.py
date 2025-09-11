import logging

from typing import Annotated

from transformers import AutoTokenizer
from langchain_chroma import Chroma
from langchain_anthropic import ChatAnthropic
from langchain.retrievers import ContextualCompressionRetriever
from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEndpoint,
    HuggingFaceEmbeddings,
)
from langchain.vectorstores import VectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnableSequence, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain.retrievers.document_compressors import (
    EmbeddingsFilter,
    DocumentCompressorPipeline,
)
from langchain_community.document_transformers import EmbeddingsRedundantFilter

from persuasion_bias.utils.prompts import RAG_PROMPT
from persuasion_bias.core.document_loaders import PersuasionDatasetLoader


class ChainRAGHuggingFace:
    """
    +------------------------------------------------+
    | Simple one-turn RAG runnable using open source |
    +------------------------------------------------+
    """

    def __init__(
        self,
        log: logging.Logger,
        model_repo_id: Annotated[str, "HuggingFace `text-generation` model"],
        embedding_model_repo_id: Annotated[str, "HuggingFace `feature-extraction` model"],
        vector_storage: VectorStore = Chroma,
    ) -> None:
        self.log = log
        self.model_repo_id = model_repo_id
        self.embedding_model_repo_id = embedding_model_repo_id
        self.vector_storage = vector_storage
        self.embedding = None
        self.model = None
        self.tokenizer = None

    def _load_models_from_hub(
        self,
    ) -> tuple[HuggingFaceEmbeddings, ChatHuggingFace, AutoTokenizer]:
        """Loads Models and Tokenizer from HuggingFace"""

        if self.embedding and self.model and self.tokenizer:
            return self.embedding, self.model, self.tokenizer

        self.log.info(msg="Loading models from HuggingFace...")
        self.embedding = HuggingFaceEmbeddings(
            model=self.embedding_model_repo_id,
            model_kwargs={"device": "cpu"},  # TODO: FAISS: --supports GPU
            encode_kwargs={"normalize_embeddings": True},
        )
        llm = HuggingFaceEndpoint(
            repo_id=self.model_repo_id,
            task="text-generation",
            max_new_tokens=512,
            do_sample=False,
            repetition_penalty=1.03,
        )
        self.model = ChatHuggingFace(llm=llm)

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.model_repo_id, trust_remote_code=True
        )
        self.log.info("Models loaded successfully")
        return self.embedding, self.model, self.tokenizer

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

    def _load_documents_from_hub(self) -> list[Document]:
        """Getting the Anthropic/persuasion dataset using custom class."""

        loader = PersuasionDatasetLoader()
        self.log.info("Loading Persuasion Dataset...")
        documents = loader.load_from_huggingface()
        self.log.info(f"Preprocessed and loaded {len(documents)} Documents.")

        return documents

    def _vectorstore_as_retriever(self) -> VectorStoreRetriever:
        """Make the Retriever using preferenced vector store."""

        documents = self._load_documents_from_hub()
        embedding, *_ = self._load_models_from_hub()
        self.log.info("Creating vector store...")
        vectorstore = self.vector_storage.from_documents(documents=documents, embedding=embedding)
        self.log.info("Vector store created.")

        return vectorstore.as_retriever(search_kwargs={"k": 3})

    def create_runnable_sequence(self) -> RunnableSequence:
        """RAG Chain. Simple one turn --no-memory."""

        _, llm, _ = self._load_models_from_hub()
        prompt = self._build_prompt()
        retriever = self._vectorstore_as_retriever()

        runnable = {"question": RunnablePassthrough(), "documents": retriever} | prompt | llm | StrOutputParser()
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
        log: logging.Logger,
        repo_id: str | None = None,
        vectorstore: VectorStore = Chroma,
        use_anthropic: bool = False,
        retriever: VectorStoreRetriever = None,
    ) -> None:
        self.log = log

        if repo_id and use_anthropic:
            error_msg = "You cannot set both `repo_id` and `use_anthropic=True`."
            self.log.error(error_msg)
            raise ValueError(error_msg)

        if repo_id:
            use_anthropic = False
            self.log.info(f"Using HuggingFace model: {repo_id}")
            llm = HuggingFaceEndpoint(repo_id=repo_id, temperature=0.1)
            self.model = ChatHuggingFace(llm=llm)

        elif use_anthropic:
            model_name = "claude-3-5-haiku-20241022"
            self.log.info(f"Using Anthropic model: {model_name}")
            self.model = ChatAnthropic(temperature=0.1, model_name=model_name)

        else:
            error_msg = "You must specify either `repo_id` or set `use_anthropic`."
            self.log.error(error_msg)
            raise ValueError(error_msg)

        self.vectorstore = vectorstore
        self._retriever = retriever

    @staticmethod
    def _load_documents_from_hub() -> list[Document]:
        """Loads Anthropic/persuasion using custom class."""

        loader = PersuasionDatasetLoader()
        documents = loader.load_from_huggingface()
        return documents

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
            self.log.info("Creating contextual compression retriever")

            self.log.debug("Loading documents")
            embedding = self._load_sentence_transformers_embeddings()
            documents = self._load_documents_from_hub()
            self.log.debug("Creating vectorstore")
            vectorstore = self.vectorstore.from_documents(documents=documents, embedding=embedding)
            self.log.debug("Setting up vectorstore retriever")
            vectorstore_retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5})

            self.log.debug("Setting up the compression pipeline")
            redundant_filter = EmbeddingsRedundantFilter(embeddings=embedding)
            relevance_filter = EmbeddingsFilter(embeddings=embedding, similarity_threshold=similarity_threshold)
            compression_pipeline = DocumentCompressorPipeline(transformers=[redundant_filter, relevance_filter])
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compression_pipeline,
                base_retriever=vectorstore_retriever,
            )
            self._retriever = compression_retriever
            self.log.info("Retriever created and cached")

        return self._retriever

    @staticmethod
    def _create_prompt() -> ChatPromptTemplate:
        """Dynamic one-turn prompt."""

        system_prompt = RAG_PROMPT

        return ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{argument}")])

    def create_runnable_sequence(self) -> RunnableSequence:
        """The full ContextualCompression RAG Chain --no-memory."""

        prompt = self._create_prompt()
        retriever = self._get_contextual_compression_retriever()
        model = self.model
        runnable = {"argument": RunnablePassthrough(), "documents": retriever} | prompt | model | StrOutputParser()

        return runnable
