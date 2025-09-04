from langchain.vectorstores import VectorStore
from langchain_chroma import Chroma
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

        ### TODO ###: optimize the system prompt
        system_prompt = """You are a helpful assistant who is an expert
        in persuassiveness and argumentation. Detect bias in the argument
        and rate it based on score. Give succinct answers, only in Markdown.
        IF ONLY the user's question resolves around argumentation,
        retrieve documents from your knowledge base. ELSE, respond
        normally as you would, without retrieval.

        Documents: {documents}
        """

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
    def _load_documents_from_hub() -> list[Document]:
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
