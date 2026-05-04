"""
Orchestration de la recherche RAG.

Responsabilite :
prendre une question utilisateur, recuperer les candidats FAISS, appliquer les
filtres, rerankers et profils optionnels, puis retourner les chunks finaux.

Pourquoi isoler ce module ?
- `rechercher()` est le coeur du retrieval, donc une brique majeure a expliquer.
- `rag.engine` peut rester le point d'entree CLI/pipeline sans porter tous les
  details FAISS, filtres et rerankers.
"""

from __future__ import annotations

from rag.config import (
    DEFAULT_TOP_K,
    DEFAULT_CANDIDATE_K,
    DEFAULT_MIN_SCORE,
    DEFAULT_CROSS_ENCODER_RERANKER,
    DEFAULT_LLM_RERANKER,
)
from rag.embeddings import creer_embeddings
from rag.env import charger_env_local
from rag.everpure import (
    prioriser_inventaires_produits as prioriser_inventaires_produits_everpure,
    question_demande_inventaire_produit as question_demande_inventaire_produit_everpure,
)
from rag.ingestion import charger_index
from rag.models import FiltreMetadata, ParametresRecherche, ResultatRecherche
from rag.profiles import profil_est_actif
from rag.query_rewrite import construire_requetes_recherche
from rag.reranking import (
    reranker_resultats,
    reranker_resultats_avec_cross_encoder,
    reranker_resultats_avec_llm,
)
from rag.retrieval import (
    diversifier_resultats_par_document,
    filtrer_resultats_pour_valeurs_techniques as filtrer_resultats_pour_valeurs_techniques_local,
    fusionner_candidats_multi_requetes,
    variable_env_booleenne,
)


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

