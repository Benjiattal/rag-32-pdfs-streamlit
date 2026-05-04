"""
Pipeline final du RAG.

Responsabilite :
enchaîner les grandes briques deja separees du projet :

1. recherche des meilleurs chunks avec `rag.search` ;
2. construction du prompt avec `rag.prompting` ;
3. appel du modele de generation avec `rag.llm`.

Pourquoi isoler ce module ?
- `rag.search` explique comment on retrouve les sources.
- `rag.prompting` explique comment on transforme ces sources en contexte.
- `rag.llm` explique comment on appelle OpenAI.
- `rag.pipeline` explique simplement comment tout cela s'enchaine.

Cette separation rend le projet plus facile a presenter : on peut montrer le
chemin complet d'une question sans replonger dans les details FAISS, BM25 ou BGE.
"""

from __future__ import annotations

import os

from rag.config import (
    DEFAULT_CANDIDATE_K,
    DEFAULT_CONTEXT_TOKEN_BUDGET,
    DEFAULT_MIN_SCORE,
    DEFAULT_TOP_K,
)
from rag.env import charger_env_local
from rag.llm import appeler_modele
from rag.models import FiltreMetadata
from rag.prompting import construire_prompt
from rag.search import rechercher


def demander(
    question: str,
    top_k: int | None = None,
    candidate_k: int | None = None,
    budget_tokens: int | None = None,
    min_score: float | None = None,
    filtre: FiltreMetadata | None = None,
    modele_generation: str | None = None,
    utiliser_query_rewrite_llm: bool | None = None,
    utiliser_reranker_llm: bool | None = None,
    utiliser_cross_encoder_reranker: bool | None = None,
) -> str:
    """
    Pipeline complet :
    question -> recherche -> prompt -> reponse OpenAI.

    Les parametres optionnels permettent de piloter le RAG depuis la CLI ou
    Streamlit. Si un parametre n'est pas fourni, on lit la variable
    d'environnement correspondante, puis la valeur par defaut du projet.
    """
    charger_env_local()

    top_k = top_k or int(os.getenv("TOP_K", str(DEFAULT_TOP_K)))
    candidate_k = candidate_k or int(os.getenv("CANDIDATE_K", str(DEFAULT_CANDIDATE_K)))
    budget_tokens = budget_tokens or int(
        os.getenv("CONTEXT_TOKEN_BUDGET", str(DEFAULT_CONTEXT_TOKEN_BUDGET))
    )
    min_score = min_score if min_score is not None else float(
        os.getenv("MIN_SCORE", str(DEFAULT_MIN_SCORE))
    )

    resultats = rechercher(
        question,
        top_k=top_k,
        candidate_k=candidate_k,
        min_score=min_score,
        filtre=filtre,
        utiliser_query_rewrite_llm=utiliser_query_rewrite_llm,
        utiliser_reranker_llm=utiliser_reranker_llm,
        utiliser_cross_encoder_reranker=utiliser_cross_encoder_reranker,
    )
    prompt = construire_prompt(question, resultats, budget_tokens=budget_tokens)
    reponse = appeler_modele(prompt, modele=modele_generation)

    return reponse
