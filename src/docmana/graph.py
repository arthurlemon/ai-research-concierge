"""LangGraph application for the DocMana AI Research Concierge."""

from __future__ import annotations

import asyncio
from typing import Dict, List

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from .prompts import ANALYZE_PROMPT, ERROR_PROMPT, GATHER_PROMPT, SYNTHESIZE_PROMPT
from .state import GraphState
from .tools import knowledge_lookup, tavily_search

# Configurable model shared across nodes
configurable_model = init_chat_model(
    configurable_fields=("model", "api_key", "max_tokens")
)


class AnalysisPlan(BaseModel):
    """Structured output for breaking down a user query."""

    sub_questions: List[str] = Field(description="Concrete sub-questions to research")


def _model_config(
    config: RunnableConfig,
    default_model: str = "openai:gpt-4.1",
    default_max_tokens: int = 10000,
) -> Dict:
    cfg = config.get("configurable", {}) if config else {}
    return {
        "model": cfg.get("model", default_model),
        "api_key": cfg.get("api_key"),
        "max_tokens": cfg.get("max_tokens", default_max_tokens),
        "tags": ["docmana"],
    }


async def analyze_query(state: GraphState, config: RunnableConfig) -> Dict:
    """Decompose the user query into sub-questions using structured output."""

    # Extract user_query from messages if not provided directly
    user_query = state.get("user_query")
    if not user_query:
        messages = state.get("messages", [])
        if messages:
            # Get the text from all messages
            user_query = get_buffer_string(messages)

    if not user_query:
        return {
            "error": "Aucune question utilisateur fournie.",
            "messages": [AIMessage(content="❌ Aucune question utilisateur fournie.")],
        }

    model = configurable_model.with_structured_output(AnalysisPlan).with_config(
        _model_config(config)
    )
    prompt = ANALYZE_PROMPT.format(user_query=user_query)

    try:
        result = await model.ainvoke([HumanMessage(content=prompt)])
        sub_questions = [q.strip(" -•\n") for q in result.sub_questions if q.strip()]
        if not sub_questions:
            sub_questions = [user_query]
        return {"user_query": user_query, "sub_questions": sub_questions}
    except Exception as exc:
        return {
            "error": f"Analyse échouée: {exc}",
            "messages": [AIMessage(content=f"❌ Analyse échouée: {exc}")],
        }


async def gather_evidence(state: GraphState, config: RunnableConfig) -> Dict:
    """Recherche des informations pour chaque sous-question via Tavily ou la base locale."""

    sub_questions = state.get("sub_questions", [])
    if not sub_questions:
        return {"error": "Aucune sous-question à traiter."}

    async def _one(sub_question: str) -> tuple[str, str | None]:
        """Recherche pour une sous-question avec fallback."""
        try:
            # Essai Tavily d'abord
            raw = await tavily_search.ainvoke({"query": sub_question}, config)

            # Si Tavily échoue ou n'a pas de clé, utiliser la base locale
            if "❌" in raw or "Aucun résultat" in raw:
                raw = await knowledge_lookup.ainvoke({"topic": sub_question}, config)

        except Exception as exc:
            # En cas d'erreur, fallback vers la base locale
            try:
                raw = await knowledge_lookup.ainvoke({"topic": sub_question}, config)
            except:  # noqa: E722
                return sub_question, f"Erreur: {exc}"

        return sub_question, raw

    pairs = await asyncio.gather(*[_one(q) for q in sub_questions])

    # Résumer les résultats bruts avec le LLM pour garder les réponses focalisées
    model = configurable_model.with_config(_model_config(config))
    evidence: Dict[str, str] = {}

    for sub_question, raw in pairs:
        # Ignorer les résultats vides ou d'erreur
        if not raw or "Aucune donnée locale" in raw or str(raw).startswith("Erreur"):
            continue

        summary_prompt = GATHER_PROMPT.format(
            sub_question=sub_question, tool_result=raw
        )
        summary = await model.ainvoke([HumanMessage(content=summary_prompt)])
        evidence[sub_question] = summary.content

    if not evidence:
        return {
            "error": "Aucune donnée exploitable retournée par les outils de recherche."
        }
    return {"tool_results": evidence}


async def synthesize_answer(state: GraphState, config: RunnableConfig) -> Dict:
    """Combine evidence into a structured final answer."""

    formatted_sub_questions = "\n".join(
        f"- {q}" for q in state.get("sub_questions", [])
    )
    formatted_evidence = "\n".join(
        f"- {sub}:\n  {text}" for sub, text in state.get("tool_results", {}).items()
    )

    prompt = SYNTHESIZE_PROMPT.format(
        user_query=state.get("user_query", ""),
        formatted_sub_questions=formatted_sub_questions,
        formatted_evidence=formatted_evidence,
    )

    model = configurable_model.with_config(
        _model_config(config, default_model="openai:gpt-4.1")
    )
    try:
        response = await model.ainvoke([HumanMessage(content=prompt)])
        return {
            "final_answer": response.content,
            "messages": [AIMessage(content=response.content)],
        }
    except Exception as exc:
        return {
            "error": f"Synthèse échouée: {exc}",
            "messages": [AIMessage(content=f"❌ Synthèse échouée: {exc}")],
        }


async def handle_error(state: GraphState, config: RunnableConfig) -> Dict:
    """Fallback node when earlier steps fail."""

    reason = state.get("error") or "Problème inconnu."
    prompt = ERROR_PROMPT.format(user_query=state.get("user_query", ""), error=reason)
    model = configurable_model.with_config(_model_config(config))
    response = await model.ainvoke([HumanMessage(content=prompt)])
    return {
        "final_answer": response.content,
        "messages": [AIMessage(content=response.content)],
    }


def _after_analyze(state: GraphState) -> str:
    if state.get("error"):
        return "error"
    if not state.get("sub_questions"):
        return "error"
    return "gather"


def _after_gather(state: GraphState) -> str:
    if state.get("error"):
        return "error"
    if not state.get("tool_results"):
        return "error"
    return "synthesize"


builder = StateGraph(GraphState)

builder.add_node("analyze_query", analyze_query)
builder.add_node("gather_evidence", gather_evidence)
builder.add_node("synthesize_answer", synthesize_answer)
builder.add_node("handle_error", handle_error)

builder.add_edge(START, "analyze_query")

builder.add_conditional_edges(
    "analyze_query",
    _after_analyze,
    {
        "gather": "gather_evidence",
        "error": "handle_error",
    },
)

builder.add_conditional_edges(
    "gather_evidence",
    _after_gather,
    {
        "synthesize": "synthesize_answer",
        "error": "handle_error",
    },
)

builder.add_edge("synthesize_answer", END)
builder.add_edge("handle_error", END)

app = builder.compile()
