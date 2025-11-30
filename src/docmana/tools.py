"""Outils de recherche pour DocMana.

Ce module contient les outils utilisés par l'agent pour collecter des informations.
"""

from __future__ import annotations

import asyncio
import os
import textwrap
from typing import Dict

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool


# Base de connaissances locale minimale pour les tests
_KB: Dict[str, str] = {
    "python vs javascript backend": textwrap.dedent(
        """
        Performance: Python excelle avec les bibliothèques CPU-bound (NumPy, Pandas) mais nécessite async pour I/O;
        JS/Node est fort pour les workloads I/O-bound avec event loop.

        Écosystème: Python dispose de frameworks web matures (Django/FastAPI); JS a Express/NestJS,
        plus alignement full-stack avec le frontend.

        Courbe d'apprentissage: Syntaxe Python conviviale pour débutants; JavaScript a des nuances
        async/event et quirks legacy des navigateurs.

        Déploiement: Tous deux se conteneurisent bien; Python se marie souvent avec serveurs WSGI/ASGI;
        apps Node se déploient en single process ou serverless handlers.

        Recrutement/communauté: Communautés larges; Python fort en data/ML, JS fort en web/front-to-back.
        """
    ),
    "cybersecurity sme": textwrap.dedent(
        """
        Ransomware ciblant les PME reste prévalent; backups et formation phishing sont des mitigations primaires.

        Adoption MFA est un quick win majeur pour réduire les risques de prise de contrôle de compte.

        Outils de détection endpoint/EDR deviennent de plus en plus abordables pour les PME.

        Services MDR (Managed Detection and Response) comblent les lacunes en personnel.

        Réglementations: NIS2 (UE) et règles sectorielles poussent les PME à adopter des contrôles de base.

        Posture cloud: mauvaises configurations (buckets S3 ouverts, IAM faible) restent des incidents courants.
        """
    ),
    "open source vs proprietary models": textwrap.dedent(
        """
        Avantages (open): transparence, self-hosting pour contrôle des données, flexibilité des coûts,
        améliorations community-driven.

        Limites (open): besoin d'infra/ops pour le serving; peut être en retard sur les modèles propriétaires
        de pointe en termes de capacités.

        Avantages (propriétaire): instruction-following de plus haute qualité, safety/guardrails intégrés,
        hébergement clé en main.

        Limites (propriétaire): vendor lock-in, préoccupations de résidence des données, coûts basés
        sur l'usage et rate limits.

        Hybride: certaines équipes prototypent avec modèles open localement et déploient du propriétaire
        pour les SLAs production.
        """
    ),
}


def _match_topic(topic: str) -> str | None:
    """Recherche dans la base de connaissances locale."""
    topic_l = topic.lower()
    for key, value in _KB.items():
        if all(token in topic_l for token in key.split()):
            return value
    # Fallback: correspondances partielles
    for key, value in _KB.items():
        if any(token in topic_l for token in key.split()):
            return value
    return None


@tool
async def knowledge_lookup(topic: str) -> str:
    """Recherche des informations dans une base de connaissances locale.

    Outil de secours pour des données curées et déterministes.
    """
    match = _match_topic(topic)
    if not match:
        return "Aucune donnée locale trouvée pour ce sujet."
    return match


@tool
async def tavily_search(query: str, config: RunnableConfig = None) -> str:
    """Recherche des informations sur le web via l'API Tavily.

    Args:
        query: La requête de recherche
        config: Configuration incluant la clé API Tavily

    Returns:
        Résultats de recherche formatés avec titres, URLs et contenu
    """
    try:
        from tavily import AsyncTavilyClient
    except ImportError:
        return "❌ Le package 'tavily-python' n'est pas installé. Installez-le avec: pip install tavily-python"

    # Récupérer la clé API
    tavily_api_key = os.environ.get("TAVILY_API_KEY")
    if not tavily_api_key:
        return "❌ TAVILY_API_KEY non trouvée dans les variables d'environnement."

    # Créer le client et effectuer la recherche
    client = AsyncTavilyClient(api_key=tavily_api_key)

    try:
        response = await asyncio.wait_for(
            client.search(query, max_results=5, include_raw_content=False), timeout=30.0
        )
    except asyncio.TimeoutError:
        return f"⏱️ Timeout lors de la recherche pour: {query}"
    except Exception as e:
        return f"❌ Erreur lors de la recherche Tavily: {str(e)}"

    # Formater les résultats
    if not response.get("results"):
        return f"Aucun résultat trouvé pour: {query}"

    formatted_output = f"Résultats de recherche pour: {query}\n\n"

    for i, result in enumerate(response["results"], 1):
        formatted_output += (
            f"\n--- SOURCE {i}: {result.get('title', 'Sans titre')} ---\n"
        )
        formatted_output += f"URL: {result.get('url', 'N/A')}\n\n"
        formatted_output += (
            f"CONTENU:\n{result.get('content', 'Pas de contenu disponible')}\n\n"
        )
        formatted_output += "-" * 80 + "\n"

    return formatted_output
