"""
==============================
SCRIPT RAG PDF - VERSION COMMENTEE
==============================

Ce fichier contient un petit RAG local pour interroger quelques PDF.

RAG signifie "Retrieval Augmented Generation".
En francais simple :

1. On lit les PDF.
2. On decoupe leur texte en petits morceaux.
3. On transforme chaque morceau en vecteur avec OpenAI.
4. On stocke ces vecteurs dans une base vectorielle FAISS.
5. Quand on pose une question, FAISS retrouve les morceaux les plus proches.
6. On donne ces morceaux au modele OpenAI pour produire une reponse.

Pourquoi ce fichier est tres commente ?
Parce que l'objectif est d'apprendre. Les commentaires expliquent autant le
"pourquoi" que le "comment".
"""

from __future__ import annotations

# ==============================
# --- 1. Imports ---
# ==============================

import argparse  # interface en ligne de commande
import os  # lecture des variables d'environnement

from rag.config import (
    PDF_DIR,
    WEB_DIR,
    INDEX_DIR,
    FAISS_INDEX_FILE,
    METADATA_FILE,
    DEFAULT_CHAT_MODEL,
    DEFAULT_CHAT_MODELS,
    DEFAULT_DOMAIN_PROFILE,
    DEFAULT_TOP_K,
    DEFAULT_CANDIDATE_K,
    DEFAULT_CONTEXT_TOKEN_BUDGET,
    DEFAULT_MIN_SCORE,
    DEFAULT_QUERY_REWRITE_LLM,
    DEFAULT_LLM_RERANKER,
    DEFAULT_CROSS_ENCODER_RERANKER,
    DEFAULT_CROSS_ENCODER_MODEL,
)
from rag.models import (
    Morceau,
    ResultatRecherche,
    FiltreMetadata,
)
from rag.llm import appeler_modele
from rag.env import charger_env_local
from rag.embeddings import (
    cle_cache_embedding,
    charger_cache_embeddings,
    sauvegarder_cache_embeddings,
    statistiques_cache_embeddings,
)
from rag.ingestion import (
    lister_pdfs,
    domaines_web_autorises,
    domaine_est_autorise,
    lister_sources_web,
    lister_documents_indexes,
    statistiques_index,
    UnpicklerCompatibleRag,
    charger_morceaux_pickle,
    charger_web_chunks,
    sauvegarder_web_chunks,
    lire_pdf,
    extraire_tableaux_page,
    convertir_tableau_en_markdown,
    extraire_extraits_techniques,
    lire_page_web,
    indexer_pages_web,
    indexer_pdf,
    charger_index,
)
from rag.retrieval import (
    variable_env_booleenne,
)
from rag.profiles import (
    lister_profils_disponibles,
)
from rag.prompting import (
    construire_prompt,
)
from rag.reranking import (
    cross_encoder_disponible,
    dernier_reranker_bge_actif,
    statistiques_cache_llm_rerank,
)
from rag.query_rewrite import (
    construire_question_recherche,
    construire_requetes_recherche,
    statistiques_cache_query_rewrite,
)
from rag.search import (
    calculer_parametres_recherche_adaptatifs,
    question_demande_portefeuille_capacites,
    question_demande_valeurs_techniques,
    rechercher,
)



# Pourquoi separer configuration, modeles et moteur ?
# -> `config.py` porte les chemins et valeurs par defaut.
# -> `models.py` porte les dataclasses partagees.
# -> `engine.py` garde l'orchestration RAG historique.


# Le chunking vit dans un module dedie : c'est une brique autonome du RAG.
# On garde ces imports dans engine.py pour conserver l'API historique.
from rag.chunking import (
    decouper_texte,
    decouper_texte_par_sections,
    extraire_sections_simples,
    ligne_parasite_pdf,
    est_titre_section_probable,
    nettoyer_titre_section,
    decouper_en_phrases,
    decouper_phrase_trop_longue,
)


# ==============================
# --- 7. Ingestion et index FAISS ---
# ==============================

# L'ingestion PDF/Web et le chargement FAISS vivent dans `rag.ingestion`.
# Les fonctions sont importees ci-dessus pour conserver l'API historique de
# `rag_pdf.py` et eviter une regression cote Streamlit/CLI.


# ==============================
# --- 8. Recherche ---
# ==============================


# Pourquoi cette etape ?
# -> On ne donne pas tous les PDF au modele.
# -> On lui donne seulement les passages les plus proches semantiquement.


# Le prompt et le contexte long vivent dans `rag.prompting`.
# On les importe plus haut pour garder l'API historique de `rag_pdf.py`.


# ==============================
# --- 12. Pipeline complet ---
# ==============================

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
    question -> recherche FAISS -> prompt -> reponse OpenAI.
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


def afficher_sources(
    question: str,
    top_k: int | None = None,
    candidate_k: int | None = None,
    min_score: float | None = None,
    filtre: FiltreMetadata | None = None,
    utiliser_query_rewrite_llm: bool | None = None,
    utiliser_reranker_llm: bool | None = None,
    utiliser_cross_encoder_reranker: bool | None = None,
) -> None:
    """Affiche les sources retrouvees pour mieux comprendre le RAG."""
    top_k = top_k or int(os.getenv("TOP_K", str(DEFAULT_TOP_K)))
    candidate_k = candidate_k or int(os.getenv("CANDIDATE_K", str(DEFAULT_CANDIDATE_K)))
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

    print("\nSources retrouvees")
    print("==================")

    if not resultats:
        print(f"Aucun resultat au-dessus du seuil MIN_SCORE={min_score:.3f}.")
        return

    for numero, resultat in enumerate(resultats, start=1):
        morceau = resultat.morceau
        if getattr(morceau, "source_type", "pdf") == "web":
            localisation = (
                f"Web : {morceau.fichier} | {getattr(morceau, 'url', '')} | "
                f"consulte le {getattr(morceau, 'date_consultation', '')}"
            )
        else:
            localisation = f"{morceau.fichier}, page {morceau.page}"

        print(
            f"{numero}. {localisation}, chunk {getattr(morceau, 'numero', '?')} | "
            f"final {resultat.score_final:.3f} | "
            f"FAISS {resultat.score_semantique:.3f} | "
            f"lexical {resultat.score_lexical:.3f}"
        )


# ==============================
# --- 13. CLI ---
# ==============================

def main() -> None:
    """
    Interface en ligne de commande.

    Exemples :

    python rag_pdf.py indexer
    python rag_pdf.py demander "Quels sont les points importants ?"
    """
    parseur = argparse.ArgumentParser(
        description="RAG documentaire avec OpenAI, FAISS et sources citees."
    )
    sous_commandes = parseur.add_subparsers(dest="commande", required=True)

    sous_commandes.add_parser(
        "indexer",
        help="Lire les PDF + sources web locales et construire l'index FAISS.",
    )

    commande_web = sous_commandes.add_parser(
        "web-indexer",
        help="Telecharger des pages web autorisees puis reconstruire l'index FAISS.",
    )
    commande_web.add_argument("urls", nargs="+", help="URLs web a indexer.")
    commande_web.add_argument(
        "--no-rebuild",
        action="store_true",
        help="Sauvegarder les pages web sans reconstruire FAISS.",
    )

    commande_demander = sous_commandes.add_parser(
        "demander",
        help="Poser une question aux PDF indexes.",
    )
    commande_demander.add_argument("question", help="Question a poser au RAG.")
    commande_demander.add_argument(
        "--sources",
        action="store_true",
        help="Afficher aussi les sources retrouvees par FAISS.",
    )
    commande_demander.add_argument(
        "--top-k",
        "--top_k",
        dest="top_k",
        type=int,
        default=None,
        help="Nombre de morceaux gardes apres reranking.",
    )
    commande_demander.add_argument(
        "--candidate-k",
        type=int,
        default=None,
        help="Nombre de candidats FAISS avant reranking.",
    )
    commande_demander.add_argument(
        "--context-tokens",
        type=int,
        default=None,
        help="Budget approximatif de tokens pour le contexte.",
    )
    commande_demander.add_argument(
        "--min-score",
        "--min_score",
        dest="min_score",
        type=float,
        default=None,
        help="Seuil minimal de score final apres reranking.",
    )
    commande_demander.add_argument(
        "--document",
        action="append",
        default=None,
        help="Filtrer sur un nom de PDF. Option repetable.",
    )
    commande_demander.add_argument(
        "--page-min",
        type=int,
        default=None,
        help="Filtrer a partir de cette page.",
    )
    commande_demander.add_argument(
        "--page-max",
        type=int,
        default=None,
        help="Filtrer jusqu'a cette page.",
    )
    commande_demander.add_argument(
        "--source-type",
        choices=["pdf", "web"],
        action="append",
        default=None,
        help="Limiter la recherche a un type de source. Option repetable.",
    )
    commande_demander.add_argument(
        "--no-query-rewrite",
        action="store_true",
        help="Desactiver la reformulation LLM pour cette question.",
    )
    commande_demander.add_argument(
        "--no-llm-reranker",
        action="store_true",
        help="Desactiver le reranking LLM pour cette question.",
    )
    commande_demander.add_argument(
        "--cross-encoder-reranker",
        action="store_true",
        help=(
            "Activer le reranker local CrossEncoder. "
            "Necessite sentence-transformers et peut telecharger le modele au premier usage."
        ),
    )

    arguments = parseur.parse_args()

    if arguments.commande == "indexer":
        indexer_pdf()
        return

    if arguments.commande == "web-indexer":
        indexer_pages_web(arguments.urls)
        if not arguments.no_rebuild:
            indexer_pdf()
        return

    if arguments.commande == "demander":
        filtre = FiltreMetadata(
            fichiers=set(arguments.document) if arguments.document else None,
            page_min=arguments.page_min,
            page_max=arguments.page_max,
            source_types=set(arguments.source_type) if arguments.source_type else None,
        )
        reponse = demander(
            arguments.question,
            top_k=arguments.top_k,
            candidate_k=arguments.candidate_k,
            budget_tokens=arguments.context_tokens,
            min_score=arguments.min_score,
            filtre=filtre,
            utiliser_query_rewrite_llm=not arguments.no_query_rewrite,
            utiliser_reranker_llm=not arguments.no_llm_reranker,
            utiliser_cross_encoder_reranker=arguments.cross_encoder_reranker,
        )

        print("\nReponse")
        print("=======")
        print(reponse)

        if arguments.sources:
            afficher_sources(
                arguments.question,
                top_k=arguments.top_k,
                candidate_k=arguments.candidate_k,
                min_score=arguments.min_score,
                filtre=filtre,
                utiliser_query_rewrite_llm=not arguments.no_query_rewrite,
                utiliser_reranker_llm=not arguments.no_llm_reranker,
                utiliser_cross_encoder_reranker=arguments.cross_encoder_reranker,
            )


if __name__ == "__main__":
    main()


# Pourquoi une CLI ?
# -> On separe l'indexation et les questions.
# -> On indexe rarement, mais on pose souvent des questions.
# -> C'est un fonctionnement standard dans les petits pipelines RAG.
