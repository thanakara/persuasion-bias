import json

from typing import Literal
from collections.abc import Callable

from langgraph.graph import END, StateGraph
from langchain.retrievers import ContextualCompressionRetriever
from langgraph.graph.state import CompiledStateGraph
from langchain.vectorstores import VectorStore
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import AIMessage, ToolMessage, SystemMessage
from langchain_core.documents import Document
from langchain_core.runnables import RunnableBinding
from langchain_core.embeddings import Embeddings
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.language_models import BaseChatModel
from langchain.retrievers.document_compressors import (
    EmbeddingsFilter,
    DocumentCompressorPipeline,
)
from langchain_community.document_transformers import EmbeddingsRedundantFilter

from persuasion_bias.core.state import GraphState, AnalysisState, BaselineState
from persuasion_bias.utils.prompts import (
    ANALYSIS_PROMPT,
    EXPLANATORY_PROMPT,
    RAG_SYSTEM_MESSAGE,
    IS_ARGUMENT_TEMPLATE,
)
from persuasion_bias.utils.wrappers import join_documents
from persuasion_bias.core.retrievers import PersuasivenessRetriever
from persuasion_bias.core.document_loaders import PersuasionDatasetLoader


class BaselinePersuasionRAG:
    def __init__(self, llm: RunnableBinding, tools_dict: dict[str, Callable]) -> None:
        self.llm = llm
        self.tools_dict = tools_dict

    def _graph_fabricate(self) -> CompiledStateGraph:
        """Builds the LangGraph workflow."""

        flow = StateGraph(BaselineState)
        flow.add_node("LLM", self.call_llm)
        flow.add_node("RetrievalAgent", self.retriever_node)

        flow.add_conditional_edges(
            source="LLM",
            path=self.should_make_retrieval,
            path_map={True: "RetrievalAgent", False: END},
        )
        flow.add_edge("RetrievalAgent", "LLM")
        flow.set_entry_point("LLM")

        return flow.compile()

    @staticmethod
    def should_make_retrieval(state: BaselineState) -> bool:
        """Decides whether to retrieve documents or END. Returns boolean."""

        *_, last_message = state.get("messages")

        return hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0

    def call_llm(self, state: BaselineState) -> BaselineState:
        """Invokes the LLM to get a response, adding it to the state."""

        messages = state.get("messages")
        content = RAG_SYSTEM_MESSAGE
        system_message = SystemMessage(content=content)
        messages_add = [system_message] + messages
        response = self.llm.invoke(messages_add)

        return {"messages": [response]}

    def retriever_node(self, state: BaselineState) -> BaselineState:
        """Node that executes the retriever tool."""

        *_, last_message = state.get("messages")
        tool_calls = last_message.tool_calls

        results = []
        for t in tool_calls:
            if t.get("name") not in self.tools_dict:
                result = "Incorrect Tool Name. Please Retry and Select from Available Tools."

            # Llama3.2 models after invocation sometimes return nested dictionaries
            try:
                assert "properties" in t.get("args")
                result = self.tools_dict.get(t.get("name"))(**t.get("args").get("properties"))
            except AssertionError:
                result = self.tools_dict.get(t.get("name"))(**t.get("args"))

            tool_message = ToolMessage(content=str(result), tool_call_id=t.get("id"))
            results.append(tool_message)

        return {"messages": results}


class BiasAnalystAgent:
    def __init__(self, llm: BaseChatModel, embedding: Embeddings, vectorstore: VectorStore) -> None:
        self.llm = llm
        self.embedding = embedding
        self.vectorstore = vectorstore
        self._retriever = None

    def _graph_fabricate(self) -> CompiledStateGraph:
        """Builds the LangGraph workflow."""

        graph = StateGraph(AnalysisState)
        graph.add_node("is_argument", self.is_argument_node)
        graph.add_node("retrieve", self.retriever_node)
        graph.add_node("analyze", self.bias_analyzer_node)
        graph.add_conditional_edges(
            "is_argument",
            self.should_continue,
            path_map={"true": "retrieve", "false": END},
        )
        graph.add_edge("retrieve", "analyze")
        graph.set_entry_point("is_argument")
        graph.set_finish_point("analyze")

        return graph.compile()

    @staticmethod
    def _load_documents_from_hub() -> list[Document]:
        """Getting the Anthropic/persuasion dataset using custom class."""

        loader = PersuasionDatasetLoader()

        return loader.load_from_huggingface()

    def _get_core_retriever(self, similarity_threshold: float = 0.55) -> ContextualCompressionRetriever:
        """
        Contextual Compression Retriever using:
            * base_retriever=PersuasivenessRetriever [Custom: `MMR` algorithm]
            * base_compressor=DocumentCompressorPipeline [redundancy -> relevance]
        """

        if self._retriever is None:
            documents = self._load_documents_from_hub()
            vectorstore = self.vectorstore.from_documents(documents=documents, embedding=self.embedding)
            vectorstore_retriever = PersuasivenessRetriever(vectorstore=vectorstore)
            redundant_filter = EmbeddingsRedundantFilter(embeddings=self.embedding)
            relevance_filter = EmbeddingsFilter(embeddings=self.embedding, similarity_threshold=similarity_threshold)
            compression_pipeline = DocumentCompressorPipeline(transformers=[redundant_filter, relevance_filter])
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compression_pipeline,
                base_retriever=vectorstore_retriever,
            )
            self._retriever = compression_retriever

        return self._retriever

    def is_argument_node(self, state: AnalysisState) -> AnalysisState:
        """Decides whether to retrieve OR end."""

        query = state.get("query")
        prompt = PromptTemplate.from_template(template=IS_ARGUMENT_TEMPLATE)
        response = self.llm.invoke(prompt.format(query=query))

        response_text = response.content.strip().lower()
        is_argument = response_text in ["true", "1", "yes"] or response_text.startswith("true")
        return {"is_argument": is_argument}

    @staticmethod
    def should_continue(state: AnalysisState) -> Literal["true", "false"]:
        """Conditional edge"""

        is_argument = state.get("is_argument")

        return "true" if is_argument else "false"

    def retriever_node(self, state: AnalysisState) -> AnalysisState:
        """Updates the state with the retrieval."""

        query = state.get("query")
        retriever = self._get_core_retriever()
        documents = retriever.invoke(query)

        content = f"Retrieved {len(documents)} documents from knowledge base."
        return {
            "messages": [AIMessage(content=content)],
            "retrieval": join_documents(documents=documents),
        }

    def bias_analyzer_node(self, state: AnalysisState) -> AnalysisState:
        """Analyzes the bias of the argument."""

        query = state.get("query")
        context = state.get("retrieval")
        prompt = PromptTemplate.from_template(template=ANALYSIS_PROMPT)

        response = self.llm.invoke(prompt.format(query=query, context=context))
        ai_message = AIMessage("Bias analysis completed.")

        try:
            analysis = json.loads(response.content)
        except json.JSONDecodeError:
            analysis = response.content

        return {"messages": [ai_message], "analysis": analysis}


class BiasExplanation:
    def __init__(self, llm: BaseChatModel, embedding: Embeddings, vectorstore: VectorStore) -> None:
        self.llm = llm
        self.embedding = embedding
        self.vectorstore = vectorstore
        # CACHE:
        self._retriever = None
        self._documents = None

    def graph_fabricate(self) -> CompiledStateGraph:
        workflow = StateGraph(GraphState)
        workflow.add_node("is_argument", self.is_argument_node)
        workflow.add_node("retrieve", self.retrieval_node)
        workflow.add_node("analyze", self.bias_analysis_node)
        workflow.add_node("explain", self.explanation_node)

        workflow.add_conditional_edges(
            source="is_argument", path=self.should_continue, path_map={"true": "retrieve", "false": END}
        )
        workflow.add_edge("retrieve", "analyze")
        workflow.add_edge("analyze", "explain")
        workflow.set_entry_point("is_argument")
        workflow.set_finish_point("explain")

        return workflow.compile(checkpointer=InMemorySaver())

    def _load_documents_from_hub(self) -> list[Document]:
        """
        Loads the Anthropic/persuasion Dataset from HuggingFace;
        preprocess it and returns a list of Documents. Caches
        """
        if self._documents is None:
            loader = PersuasionDatasetLoader()
            self._documents = loader.load_from_huggingface()

        return self._documents

    def get_core_retriever(self, similarity_threshold: float = 0.55) -> ContextualCompressionRetriever:
        """Builds the Contextual Compression Retriever. Caches"""

        if self._retriever is None:
            documents = self._load_documents_from_hub()
            vectorstore = self.vectorstore.from_documents(documents=documents, embedding=self.embedding)
            vectorstore_retriever = PersuasivenessRetriever(vectorstore=vectorstore)
            redundant_filter = EmbeddingsRedundantFilter(embeddings=self.embedding)
            relevance_filter = EmbeddingsFilter(embeddings=self.embedding, similarity_threshold=similarity_threshold)
            compression_pipeline = DocumentCompressorPipeline(transformers=[redundant_filter, relevance_filter])
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compression_pipeline, base_retriever=vectorstore_retriever
            )
            self._retriever = compression_retriever

        return self._retriever

    def is_argument_node(self, state: GraphState) -> GraphState:
        """Decides whether to retrieve OR __end__."""

        query = state.get("query")
        prompt = PromptTemplate.from_template(template=IS_ARGUMENT_TEMPLATE)
        response = self.llm.invoke(prompt.format(query=query))

        response_text = response.content.strip().lower()
        is_argument = response_text in ["true", "1", "yes"] or response_text.startswith("true")

        return {"is_argument": is_argument, "messages": [AIMessage(content=response.content)]}

    @staticmethod
    def should_continue(state: GraphState) -> Literal["true", "false"]:
        """Conditional edge"""

        is_argument = state.get("is_argument")

        return "true" if is_argument else "false"

    def retrieval_node(self, state: GraphState) -> GraphState:
        """Updates the state with retrieval from knowledge base."""

        query = state.get("query")
        retriever = self.get_core_retriever()
        retrieved_documents = retriever.invoke(query)
        content = f"Retrieved {len(retrieved_documents)} documents from knowledge base"

        return {"messages": [AIMessage(content=content)], "retrieval": join_documents(documents=retrieved_documents)}

    def bias_analysis_node(self, state: GraphState) -> GraphState:
        """Gives a bias analysis of the argument."""

        query = state.get("query")
        context = state.get("retrieval")
        prompt = PromptTemplate.from_template(template=ANALYSIS_PROMPT)

        response = self.llm.invoke(prompt.format(query=query, context=context))
        message = AIMessage(content="I have completed the analysis")

        try:
            analysis = json.dumps(response.content)
        except json.JSONDecodeError:
            analysis = response.content

        return {"messages": [message], "analysis": analysis}

    def explanation_node(self, state: GraphState) -> GraphState:
        """Based on analysis final bias explanation."""

        query = state.get("query")
        analysis = state.get("analysis")
        prompt = PromptTemplate.from_template(template=EXPLANATORY_PROMPT)

        explanation = self.llm.invoke(prompt.format(query=query, analysis=analysis))
        message = AIMessage(content="I have generated a bias explanation for you")

        return {"explanation": explanation.content, "messages": [message]}
