"""Application FastAPI pour DocMana AI Research Concierge."""

from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .graph import app as graph_app


# Modèles de requête/réponse
class QueryRequest(BaseModel):
    """Modèle de requête pour les questions de recherche."""

    query: str = Field(
        ..., description="Question de recherche de l'utilisateur", min_length=1
    )
    model: Optional[str] = Field(
        default="openai:gpt-4.1",
        description="Modèle LLM à utiliser (ex: 'openai:gpt-4.1', 'anthropic:claude-3-5-sonnet-20241022')",
    )
    max_tokens: Optional[int] = Field(
        default=10000,
        description="Nombre maximum de tokens pour les réponses LLM",
        ge=100,
        le=100000,
    )


class QueryResponse(BaseModel):
    """Modèle de réponse pour les questions de recherche."""

    user_query: str = Field(..., description="La question originale de l'utilisateur")
    sub_questions: List[str] = Field(
        default_factory=list, description="Sous-questions décomposées"
    )
    tool_results: Dict[str, str] = Field(
        default_factory=dict, description="Preuves collectées pour chaque sous-question"
    )
    final_answer: Optional[str] = Field(
        None, description="La réponse finale synthétisée"
    )
    error: Optional[str] = Field(
        None, description="Message d'erreur si la requête a échoué"
    )


class HealthResponse(BaseModel):
    """Réponse du health check."""

    status: str
    service: str


# Initialisation de l'application FastAPI
api = FastAPI(
    title="DocMana AI Research Concierge",
    description="Assistant de recherche IA basé sur LangGraph qui décompose les questions, collecte des preuves et synthétise les réponses",
    version="0.1.0",
)

# Ajout du middleware CORS
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # À configurer de manière appropriée pour la production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@api.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Endpoint de vérification de santé.

    Retourne le statut du service pour vérifier qu'il fonctionne correctement.
    """
    return HealthResponse(status="healthy", service="docmana")


@api.post("/research", response_model=QueryResponse)
async def research_query(request: QueryRequest):
    """
    Exécute une requête de recherche via le flow d'agent DocMana.

    L'agent va :
    1. Analyser et décomposer votre question en sous-questions
    2. Collecter des preuves depuis la recherche web (Tavily) et/ou la base de connaissances locale
    3. Synthétiser les preuves en une réponse complète

    Args:
        request: QueryRequest contenant la question utilisateur et la configuration optionnelle

    Returns:
        QueryResponse avec les questions décomposées, les preuves et la réponse finale

    Raises:
        HTTPException: Si le traitement de la requête échoue
    """
    try:
        # Préparation de l'état initial
        initial_state = {
            "messages": [],
            "user_query": request.query,
            "sub_questions": [],
            "tool_results": {},
            "final_answer": None,
            "error": None,
        }

        # Préparation de la configuration avec remplacement optionnel du modèle
        config = {
            "configurable": {
                "model": request.model,
                "max_tokens": request.max_tokens,
            }
        }

        # Invocation de l'agent LangGraph
        result = await graph_app.ainvoke(initial_state, config=config)

        # Retour de la réponse
        return QueryResponse(
            user_query=result.get("user_query", request.query),
            sub_questions=result.get("sub_questions", []),
            tool_results=result.get("tool_results", {}),
            final_answer=result.get("final_answer"),
            error=result.get("error"),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Échec du traitement de la requête : {str(e)}"
        )


@api.get("/")
async def root():
    """
    Endpoint racine avec les informations de l'API.

    Fournit un aperçu du service et des endpoints disponibles.
    """
    return {
        "service": "DocMana AI Research Concierge",
        "version": "0.1.0",
        "description": "Assistant de recherche IA basé sur LangGraph",
        "endpoints": {
            "health": "GET /health - Vérification de santé du service",
            "research": "POST /research - Exécuter une requête de recherche",
            "docs": "GET /docs - Documentation interactive (Swagger UI)",
            "redoc": "GET /redoc - Documentation alternative (ReDoc)",
        },
    }
