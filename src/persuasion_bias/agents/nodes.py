import logging

from typing import Literal

from langchain_core.prompts import PromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.language_models import BaseChatModel

from persuasion_bias.utils import get_last_message
from persuasion_bias.schemas.state import GraphState
from persuasion_bias.schemas.models import BiasAnalysis, ArgumentClassification

logger = logging.getLogger(__name__)


def classify_node(state: GraphState, model: BaseChatModel, prompt: str) -> dict:
    query = get_last_message(state, "messages").content
    result: ArgumentClassification = (
        model
        .with_structured_output(ArgumentClassification)
        .invoke(
            [
                SystemMessage(content=prompt),
                HumanMessage(content=query),
            ]
        )
    )  # fmt: skip
    logger.info("classify_node: is_argument=%s", result.is_argument)
    branch_msg = HumanMessage(content=query)

    return {
        "query": query,
        "is_argument": result.is_argument,
        ("conversation_messages", "analysis_messages")[result.is_argument]: [branch_msg],
        "messages": [branch_msg],
    }


def route_classify(state: GraphState) -> Literal["true", "false"]:
    return "true" if state.get("is_argument") else "false"


def conversation_llm_node(state: GraphState, model: BaseChatModel, prompt: str) -> dict:
    response = model.invoke([SystemMessage(content=prompt)] + state["conversation_messages"])

    logger.info("conversation_llm: tool_calls=%s", [tc["name"] for tc in (response.tool_calls or [])])
    return {"messages": [response], "conversation_messages": [response]}


def retrieval_node(state: GraphState, model: BaseChatModel, prompt: str) -> dict:
    response = model.invoke([SystemMessage(content=prompt)] + state["analysis_messages"])
    logger.info("retrieval_node: tool_calls=%s", [tc["name"] for tc in (response.tool_calls or [])])
    return {"messages": [response], "analysis_messages": [response]}


def analyze_node(state: GraphState, model: BaseChatModel, prompt: str) -> dict:
    retrieval_msg = get_last_message(state, "analysis_messages")
    query = state["query"]
    context = retrieval_msg.content

    prompt = PromptTemplate.from_template(prompt)

    format_instructions = JsonOutputParser(pydantic_object=BiasAnalysis).get_format_instructions()

    fmt_prompt = prompt.format(
        query=query,
        context=context,
        format_instructions=format_instructions,
    )
    analysis: BiasAnalysis = model.with_structured_output(BiasAnalysis).invoke(fmt_prompt)
    logger.info("analyze_node: overall_bias_score=%.2f", analysis.overall_bias_score)

    return {"analysis": analysis}


def explain_node(state: GraphState, model: BaseChatModel, prompt: str) -> dict:
    user_choice = state.get("user_choice")
    logger.info("explain_node: user_choice=%s", user_choice)
    analysis: BiasAnalysis = state["analysis"].model_dump_json(indent=2)

    if user_choice == "n":
        msg = AIMessage(content=analysis)
        return {"messages": [msg], "analysis_messages": [msg]}

    query = state["query"]

    prompt = PromptTemplate.from_template(prompt)
    ai_msg = model.invoke(prompt.format(query=query, analysis=analysis))

    return {"messages": [ai_msg], "explanation": ai_msg.content, "analysis_messages": [ai_msg]}
