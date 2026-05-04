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
import hashlib  # cree une empreinte stable pour le cache d'embeddings
import os  # lecture des variables d'environnement
import pickle  # sauvegarde / chargement d'objets Python

from rag.config import (
    BASE_DIR,
    PDF_DIR,
    INDEX_DIR,
    WEB_DIR,
    FAISS_INDEX_FILE,
    METADATA_FILE,
    WEB_CHUNKS_FILE,
    EMBEDDING_CACHE_FILE,
    QUERY_REWRITE_CACHE_FILE,
    LLM_RERANK_CACHE_FILE,
    DEFAULT_CHAT_MODEL,
    DEFAULT_CHAT_MODELS,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_TOP_K,
    DEFAULT_CANDIDATE_K,
    DEFAULT_CONTEXT_TOKEN_BUDGET,
    DEFAULT_CHUNK_SIZE_CHARS,
    DEFAULT_CHUNK_MIN_CHARS,
    DEFAULT_CHUNK_OVERLAP_SENTENCES,
    DEFAULT_MIN_SCORE,
    DEFAULT_TEMPERATURE,
    DEFAULT_ALLOWED_WEB_DOMAINS,
    DEFAULT_QUERY_REWRITE_LLM,
    DEFAULT_LLM_RERANKER,
    DEFAULT_LLM_RERANK_CANDIDATES,
    DEFAULT_CROSS_ENCODER_RERANKER,
    DEFAULT_CROSS_ENCODER_MODEL,
    DEFAULT_CROSS_ENCODER_RERANK_CANDIDATES,
    DEFAULT_BGE_WORKER_TIMEOUT_SECONDS,
    QUERY_REWRITE_PROMPT_VERSION,
    LLM_RERANK_PROMPT_VERSION,
    POIDS_SCORE_SEMANTIQUE,
    POIDS_SCORE_BM25,
)
from rag.models import (
    Morceau,
    ResultatRecherche,
    FiltreMetadata,
    ParametresRecherche,
)
from rag.llm import appeler_modele, creer_client_openai
from rag.env import charger_env_local
from rag.embeddings import (
    cle_cache_embedding,
    charger_cache_embeddings,
    sauvegarder_cache_embeddings,
    statistiques_cache_embeddings,
    creer_embeddings,
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
    normaliser_texte_recherche,
    tokeniser_recherche,
    normaliser_mots,
    score_lexical,
    scores_bm25,
    enrichir_question_pour_recherche,
    ajouter_requete_unique,
    construire_question_recherche as construire_question_recherche_locale,
    construire_requetes_recherche as construire_requetes_recherche_locales,
    fusionner_candidats_multi_requetes,
    diversifier_resultats_par_document,
    contient_valeurs_techniques,
    filtrer_resultats_pour_valeurs_techniques as filtrer_resultats_pour_valeurs_techniques_local,
)
from rag.profiles import (
    lister_profils_disponibles,
    profil_domaine_actif,
    profil_est_actif,
)
from rag.everpure import (
    prioriser_inventaires_produits as prioriser_inventaires_produits_everpure,
    question_demande_inventaire_produit as question_demande_inventaire_produit_everpure,
)
from rag.prompting import (
    construire_contexte_long,
    construire_prompt,
    estimer_tokens,
)
from rag.reranking import (
    charger_cache_llm_rerank,
    charger_cross_encoder,
    cle_cache_llm_rerank,
    cross_encoder_disponible,
    dernier_reranker_bge_actif,
    extraire_liste_json_entiers,
    reranker_resultats,
    reranker_resultats_avec_cross_encoder,
    reranker_resultats_avec_llm,
    sauvegarder_cache_llm_rerank,
    statistiques_cache_llm_rerank,
)
from rag.query_rewrite import (
    charger_cache_query_rewrite,
    cle_cache_query_rewrite,
    construire_question_recherche,
    construire_requetes_recherche,
    reecrire_question_avec_llm,
    sauvegarder_cache_query_rewrite,
    statistiques_cache_query_rewrite,
)



# Pourquoi separer configuration, modeles et moteur ?
# -> `config.py` porte les chemins et valeurs par defaut.
# -> `models.py` porte les dataclasses partagees.
# -> `engine.py` garde l'orchestration RAG historique.


def variable_env_booleenne(nom: str, valeur_defaut: bool) -> bool:
    """
    Lit une variable d'environnement comme un booléen.

    Exemples acceptés :
    - vrai : "1", "true", "yes", "oui", "on"
    - faux : "0", "false", "no", "non", "off"

    C'est pratique pour activer/désactiver des options sans modifier le code.
    """
    valeur = os.getenv(nom)

    if valeur is None:
        return valeur_defaut

    return valeur.strip().lower() in {"1", "true", "yes", "oui", "on"}


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


def question_demande_valeurs_techniques(question: str) -> bool:
    """Detecte les questions qui demandent des chiffres/specifications."""
    question_minuscule = question.lower()
    indices = [
        "liste",
        "consommation",
        "consommations",
        "electrique",
        "électrique",
        "watts",
        "watt",
        "puissance",
        "spec",
        "specification",
        "spécification",
        "capacite",
        "capacité",
        "capacites",
        "capacités",
        "volumetrie",
        "volumetries",
        "volumétrie",
        "volumétries",
        "capacity",
        "capacities",
        "raw",
        "effective",
        "usable",
        "dimensions",
    ]
    return any(indice in question_minuscule for indice in indices)


def question_demande_portefeuille_capacites(question: str) -> bool:
    """
    Detecte les questions larges qui demandent des capacites sur un portefeuille.

    Dans ce cas, on veut eviter que les 8 meilleurs chunks viennent tous du meme
    document. On prefere une vue plus diversifiee entre plusieurs sources.
    """
    question_minuscule = question.lower()
    mots_portefeuille = ["toutes", "tous", "portfolio", "portefeuille", "solutions"]
    mots_capacite = [
        "volumetrie",
        "volumetries",
        "volumétrie",
        "volumétries",
        "capacite",
        "capacites",
        "capacité",
        "capacités",
        "capacity",
        "capacities",
    ]

    return any(mot in question_minuscule for mot in mots_portefeuille) and any(
        mot in question_minuscule for mot in mots_capacite
    )


def calculer_parametres_recherche_adaptatifs(
    question: str,
    top_k: int,
    candidate_k: int,
    min_score: float,
) -> ParametresRecherche:
    """
    Ajuste automatiquement la largeur de recherche selon la question.

    Principe suivi :
    - top_k = nombre de chunks donnés au modèle final ;
    - candidate_k = nombre de chunks récupérés avant reranking ;
    - min_score = seuil minimal après reranking.

    Pourquoi adapter ?
    Une question très ciblée n'a pas le même besoin qu'une question large.
    Si top_k est trop petit sur une question large, le modèle n'a tout simplement
    pas assez de contexte. S'il est trop grand sur une question simple, on ajoute
    du bruit.
    """
    question_minuscule = question.lower()
    raisons: list[str] = []

    top_k_effectif = top_k
    candidate_k_effectif = max(candidate_k, top_k)
    min_score_effectif = min_score

    if profil_est_actif("everpure") and question_demande_inventaire_produit_everpure(
        question
    ):
        top_k_effectif = max(top_k_effectif, 12)
        candidate_k_effectif = max(candidate_k_effectif, 180)
        min_score_effectif = min(min_score_effectif, 0.0)
        raisons.append("profil everpure : recherche elargie pour inventaire produit")

    elif question_demande_portefeuille_capacites(question):
        top_k_effectif = max(top_k_effectif, 12)
        candidate_k_effectif = max(candidate_k_effectif, 160)
        min_score_effectif = min(min_score_effectif, 0.0)
        raisons.append("question portefeuille/capacités : recherche très élargie")

    elif question_demande_valeurs_techniques(question):
        top_k_effectif = max(top_k_effectif, 8)
        candidate_k_effectif = max(candidate_k_effectif, 120)
        min_score_effectif = min(min_score_effectif, 0.0)
        raisons.append("question technique : recherche élargie et seuil abaissé")

    elif any(
        mot in question_minuscule
        for mot in [
            "liste",
            "lister",
            "toutes",
            "tous",
            "compare",
            "comparaison",
            "synthese",
            "synthèse",
            "avantages",
            "inconvenients",
            "inconvénients",
            "resume",
            "résume",
        ]
    ):
        top_k_effectif = max(top_k_effectif, 10)
        candidate_k_effectif = max(candidate_k_effectif, 120)
        min_score_effectif = min(min_score_effectif, 0.05)
        raisons.append("question de synthèse/liste : contexte élargi")

    else:
        candidate_k_effectif = max(candidate_k_effectif, 60)
        raisons.append("question standard : paramètres utilisateur conservés")

    # Dernière sécurité : le reranking a besoin de plus de candidats que de
    # résultats finaux, sinon il ne peut presque rien améliorer.
    candidate_k_effectif = max(candidate_k_effectif, top_k_effectif * 8)

    return ParametresRecherche(
        top_k=top_k_effectif,
        candidate_k=candidate_k_effectif,
        min_score=min_score_effectif,
        raison="; ".join(raisons),
    )


def filtrer_resultats_pour_valeurs_techniques(
    question: str,
    resultats: list[ResultatRecherche],
) -> list[ResultatRecherche]:
    """
    Wrapper historique qui injecte la detection d'intention technique.

    Le filtrage lui-meme vit dans `rag.retrieval`; la detection metier reste ici
    pour l'instant afin de limiter le deplacement des heuristiques.
    """
    return filtrer_resultats_pour_valeurs_techniques_local(
        question,
        resultats,
        question_demande_valeurs_techniques_fn=question_demande_valeurs_techniques,
    )



def rechercher(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    min_score: float = DEFAULT_MIN_SCORE,
    filtre: FiltreMetadata | None = None,
    adaptatif: bool = True,
    utiliser_query_rewrite_llm: bool | None = None,
    utiliser_reranker_llm: bool | None = None,
    utiliser_cross_encoder_reranker: bool | None = None,
) -> list[ResultatRecherche]:
    """
    Recherche les morceaux les plus pertinents pour une question.
    """
    charger_env_local()

    if adaptatif:
        parametres = calculer_parametres_recherche_adaptatifs(
            question=question,
            top_k=top_k,
            candidate_k=candidate_k,
            min_score=min_score,
        )
        top_k = parametres.top_k
        candidate_k = parametres.candidate_k
        min_score = parametres.min_score

    index, morceaux = charger_index()
    filtre = filtre or FiltreMetadata()

    # On construit plusieurs requetes de retrieval :
    # - la question originale ;
    # - une reformulation LLM courte ;
    # - une expansion deterministe avec synonymes et acronymes.
    #
    # Chaque requete interroge FAISS separement. On fusionne ensuite les chunks.
    # C'est plus propre qu'une seule requete geante, car chaque strategie garde
    # son signal propre.
    requetes_recherche = construire_requetes_recherche(
        question,
        utiliser_query_rewrite_llm=utiliser_query_rewrite_llm,
    )
    textes_requetes = [requete for _, requete in requetes_recherche]
    vecteurs_questions = creer_embeddings(textes_requetes)
    question_recherche = "\n\n".join(
        f"{libelle} : {requete}" for libelle, requete in requetes_recherche
    )

    # On recupere plus de candidats que le nombre final voulu.
    # C'est indispensable pour permettre le reranking et les filtres metadata.
    nombre_total = index.ntotal
    nombre_candidats = min(nombre_total, max(top_k, candidate_k))

    if (
        filtre.fichiers
        or filtre.page_min is not None
        or filtre.page_max is not None
        or filtre.source_types
    ):
        nombre_candidats = nombre_total

    resultats_par_requete: list[tuple[str, list[tuple[float, int]]]] = []

    for (libelle, _), vecteur_question in zip(requetes_recherche, vecteurs_questions):
        scores, indices = index.search(vecteur_question.reshape(1, -1), nombre_candidats)
        resultats_par_requete.append((libelle, list(zip(scores[0], indices[0]))))

    candidats = fusionner_candidats_multi_requetes(
        resultats_par_requete=resultats_par_requete,
        morceaux=morceaux,
        filtre=filtre,
    )

    tous_les_resultats_rerankes = reranker_resultats(question_recherche, candidats)
    resultats_rerankes = [
        resultat for resultat in tous_les_resultats_rerankes if resultat.score_final >= min_score
    ]
    resultats_rerankes = filtrer_resultats_pour_valeurs_techniques(question, resultats_rerankes)

    if profil_est_actif("everpure"):
        resultats_rerankes = prioriser_inventaires_produits_everpure(
            question,
            resultats_rerankes,
        )

    if utiliser_cross_encoder_reranker is None:
        utiliser_cross_encoder_reranker = variable_env_booleenne(
            "RAG_CROSS_ENCODER_RERANKER",
            DEFAULT_CROSS_ENCODER_RERANKER,
        )

    if utiliser_cross_encoder_reranker:
        resultats_rerankes = reranker_resultats_avec_cross_encoder(
            question=question,
            resultats=resultats_rerankes,
            top_k=top_k,
        )

    if utiliser_reranker_llm is None:
        utiliser_reranker_llm = variable_env_booleenne(
            "RAG_LLM_RERANKER",
            DEFAULT_LLM_RERANKER,
        )

    if utiliser_reranker_llm:
        resultats_rerankes = reranker_resultats_avec_llm(
            question=question,
            resultats=resultats_rerankes,
            top_k=top_k,
        )

    if question_demande_portefeuille_capacites(question):
        resultats_rerankes = diversifier_resultats_par_document(
            resultats_rerankes,
            top_k=top_k,
            max_par_document=2,
        )

    return resultats_rerankes[:top_k]


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
