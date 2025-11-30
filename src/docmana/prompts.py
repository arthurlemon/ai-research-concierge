"""Modèles de prompts pour les nœuds DocMana."""

ANALYZE_PROMPT = """
Tu es un assistant de recherche. Analyse la question de l'utilisateur et détermine comment y répondre.

Question de l'utilisateur : {user_query}

Instructions :
- Si la question est simple et directe, retourne-la telle quelle ou légèrement reformulée
- Si la question est complexe ou nécessite plusieurs aspects, décompose-la en 2-5 sous-questions concrètes et indépendantes
- Chaque sous-question doit être spécifique et recherchable
- Évite les duplications et les questions trop génériques

Réponds uniquement avec une liste de questions (une par ligne, précédée de "-").
"""

GATHER_PROMPT = """
Tu collectes des preuves pour une sous-question de recherche.
Extrais les faits clés pertinents des résultats de recherche fournis.

Sous-question : {sub_question}
Résultats de recherche :
{tool_result}

Retourne une liste concise des faits les plus pertinents sous forme de bullet points.
"""

SYNTHESIZE_PROMPT = """
Tu es un assistant de recherche qui synthétise des résultats.

Question de l'utilisateur : {user_query}

Sous-questions traitées :
{formatted_sub_questions}

Résultats par sous-question :
{formatted_evidence}

Rédige une réponse structurée avec cette forme :

# [Titre basé sur la question]

## Résumé exécutif
- 2-4 points clés synthétisant les principales conclusions

## Détails par sous-thème
- Développe chaque sous-question avec les résultats obtenus
- Utilise des paragraphes courts ou des bullet points

## Limites et incertitudes
- Mentionne les lacunes dans les données
- Indique les hypothèses faites
- Signale les informations manquantes

Reste factuel et concis. Utilise un ton professionnel.
"""

ERROR_PROMPT = """
Nous n'avons pas pu rassembler suffisamment d'informations pour répondre à : {user_query}

Raison : {error}

Propose une question de clarification à l'utilisateur OU suggère un angle plus précis pour la recherche.
Sois constructif et aide l'utilisateur à reformuler sa demande.
"""
