"""
Recherche documentaire : briques lexicales pures.

Responsabilite :
regrouper les fonctions deterministes utilisees par le retrieval hybride.

Ce module ne charge ni FAISS, ni OpenAI. Il doit rester rapide a importer et
facile a tester. Les fonctions plus sensibles comme `rechercher()` restent pour
l'instant dans `rag.engine`.
"""

from __future__ import annotations

import math
import os
import re
import unicodedata
from collections.abc import Callable

from rag.config import DEFAULT_QUERY_REWRITE_LLM
from rag.models import FiltreMetadata, Morceau, ResultatRecherche
from rag.profiles import profil_domaine_actif


def normaliser_texte_recherche(texte: str) -> str:
    """
    Prepare un texte pour la recherche lexicale.

    Exemple : "électrique" devient "electrique".
    Cela evite de rater un match lexical a cause des accents.
    """
    texte = unicodedata.normalize("NFKD", texte)
    texte = texte.encode("ascii", "ignore").decode("ascii")
    return texte.lower()


def tokeniser_recherche(texte: str) -> list[str]:
    """Transforme un texte en mots utiles pour BM25 et le debug lexical."""
    texte = normaliser_texte_recherche(texte)
    return re.findall(r"[a-zA-Z0-9_/-]{3,}", texte)


def normaliser_mots(texte: str) -> set[str]:
    """
    Extrait des mots utiles pour un reranking lexical simple.

    Principe :
    - FAISS retrouve les passages proches par le sens.
    - Le reranking lexical ajoute un bonus aux passages qui reprennent les mots
      importants de la question.
    """
    mots_vides = {
        "avec",
        "dans",
        "des",
        "du",
        "elle",
        "est",
        "les",
        "leur",
        "leurs",
        "mais",
        "par",
        "pas",
        "plus",
        "pour",
        "que",
        "qui",
        "quoi",
        "sont",
        "sur",
        "une",
        "aux",
        "ces",
        "ses",
        "the",
        "and",
        "for",
        "with",
    }
    mots = tokeniser_recherche(texte)
    return {mot for mot in mots if mot not in mots_vides}


def score_lexical(question: str, texte: str) -> float:
    """
    Calcule un petit score lexical entre 0 et 1.

    Ce score n'est pas une intelligence semantique. Il verifie seulement si les
    mots importants de la question apparaissent dans le passage.
    """
    mots_question = normaliser_mots(question)

    if not mots_question:
        return 0.0

    mots_texte = normaliser_mots(texte)
    mots_communs = mots_question.intersection(mots_texte)

    return len(mots_communs) / len(mots_question)


def scores_bm25(question: str, textes: list[str]) -> list[float]:
    """
    Calcule un score BM25 simplifie pour chaque texte candidat.

    BM25 est une methode classique des moteurs de recherche :
    - elle favorise les passages qui contiennent les mots importants ;
    - elle evite qu'un mot repete 30 fois domine tout ;
    - elle tient compte de la longueur des passages.

    Ici on l'utilise uniquement sur les candidats deja trouves par FAISS.
    C'est donc peu couteux, mais souvent plus robuste qu'un simple comptage.
    """
    mots_question = list(normaliser_mots(question))

    if not mots_question or not textes:
        return [0.0 for _ in textes]

    documents = [tokeniser_recherche(texte) for texte in textes]
    longueurs = [len(document) for document in documents]
    longueur_moyenne = sum(longueurs) / max(1, len(longueurs))

    frequence_documents: dict[str, int] = {}
    for document in documents:
        mots_uniques = set(document)
        for mot in mots_question:
            if mot in mots_uniques:
                frequence_documents[mot] = frequence_documents.get(mot, 0) + 1

    nombre_documents = len(documents)
    k1 = 1.5
    b = 0.75
    scores: list[float] = []

    for document, longueur_document in zip(documents, longueurs):
        frequences: dict[str, int] = {}
        for mot in document:
            frequences[mot] = frequences.get(mot, 0) + 1

        score = 0.0
        for mot in mots_question:
            frequence = frequences.get(mot, 0)
            if frequence == 0:
                continue

            df = frequence_documents.get(mot, 0)
            idf = math.log(1 + (nombre_documents - df + 0.5) / (df + 0.5))
            normalisation_longueur = 1 - b + b * (
                longueur_document / max(1, longueur_moyenne)
            )
            score += idf * (frequence * (k1 + 1)) / (
                frequence + k1 * normalisation_longueur
            )

        scores.append(score)

    score_max = max(scores) if scores else 0.0
    if score_max <= 0:
        return [0.0 for _ in scores]

    # Normalisation entre 0 et 1 pour combiner proprement avec le score FAISS.
    return [score / score_max for score in scores]


def enrichir_question_pour_recherche(question: str) -> str:
    """
    Applique l'expansion deterministe du profil actif.

    Par defaut, le profil `generic` ne fait aucune hypothese metier sur les PDF.
    Les synonymes specifiques a un corpus vivent dans `rag/domain_profiles/*.json`
    et s'activent via `RAG_DOMAIN_PROFILE=<nom_du_profil>`.
    """
    return profil_domaine_actif().expand_question(question)



def variable_env_booleenne(nom: str, valeur_defaut: bool) -> bool:
    """
    Lit une variable d'environnement comme un booleen.

    Cette petite fonction est dupliquee ici volontairement pour garder
    `rag.retrieval` independant de `rag.engine` et eviter un import circulaire.
    """
    valeur = os.getenv(nom)

    if valeur is None:
        return valeur_defaut

    return valeur.strip().lower() in {"1", "true", "yes", "oui", "on"}


def construire_question_recherche(
    question: str,
    utiliser_query_rewrite_llm: bool | None = None,
    rewrite_fn: Callable[[str], str] | None = None,
) -> str:
    """
    Construit la question réellement envoyée aux embeddings.

    Elle combine deux approches :
    1. Query rewriting LLM : dynamique, utile pour fautes, synonymes, acronymes.
    2. Expansion déterministe : stable, maîtrisée, adaptée à notre corpus.

    `rewrite_fn` est injectee par `rag.engine`, car l'appel OpenAI reste dans le
    moteur historique pour le moment. Sans `rewrite_fn`, la fonction reste 100 %
    locale et utilise seulement l'expansion deterministe.
    """
    if utiliser_query_rewrite_llm is None:
        utiliser_query_rewrite_llm = variable_env_booleenne(
            "RAG_QUERY_REWRITE_LLM",
            DEFAULT_QUERY_REWRITE_LLM,
        )

    question_regles = enrichir_question_pour_recherche(question)
    rewrite_llm = ""

    if utiliser_query_rewrite_llm and rewrite_fn is not None:
        rewrite_llm = rewrite_fn(question)

    if not rewrite_llm:
        return question_regles

    return (
        f"{question}\n\n"
        f"Reformulation LLM pour recherche : {rewrite_llm}\n\n"
        f"{question_regles}"
    )


def ajouter_requete_unique(
    requetes: list[tuple[str, str]],
    libelle: str,
    requete: str,
) -> None:
    """
    Ajoute une requete de recherche seulement si elle apporte du contenu nouveau.

    En multi-requetes, on veut eviter d'envoyer trois fois la meme question a
    FAISS. Cela economise des embeddings, reduit le bruit et rend le debug plus
    lisible.
    """
    requete = " ".join(requete.split())

    if not requete:
        return

    requete_normalisee = normaliser_texte_recherche(requete)
    deja_presentes = {
        normaliser_texte_recherche(requete_existante)
        for _, requete_existante in requetes
    }

    if requete_normalisee in deja_presentes:
        return

    requetes.append((libelle, requete))


def construire_requetes_recherche(
    question: str,
    utiliser_query_rewrite_llm: bool | None = None,
    rewrite_fn: Callable[[str], str] | None = None,
) -> list[tuple[str, str]]:
    """
    Construit plusieurs requetes de retrieval au lieu d'une seule grosse requete.

    Requetes produites :
    1. question utilisateur originale ;
    2. reformulation LLM courte, si active et si `rewrite_fn` est fournie ;
    3. expansion deterministe avec synonymes et acronymes metier.
    """
    if utiliser_query_rewrite_llm is None:
        utiliser_query_rewrite_llm = variable_env_booleenne(
            "RAG_QUERY_REWRITE_LLM",
            DEFAULT_QUERY_REWRITE_LLM,
        )

    requetes: list[tuple[str, str]] = []
    ajouter_requete_unique(requetes, "question originale", question)

    if utiliser_query_rewrite_llm and rewrite_fn is not None:
        rewrite_llm = rewrite_fn(question)
        ajouter_requete_unique(requetes, "reformulation LLM", rewrite_llm)

    question_enrichie = enrichir_question_pour_recherche(question)
    ajouter_requete_unique(requetes, "expansion synonymes", question_enrichie)

    return requetes


def fusionner_candidats_multi_requetes(
    resultats_par_requete: list[tuple[str, list[tuple[float, int]]]],
    morceaux: list[Morceau],
    filtre: FiltreMetadata,
) -> list[tuple[float, Morceau]]:
    """
    Fusionne les candidats FAISS issus de plusieurs requetes.

    Chaque recherche FAISS retourne des indices de chunks. Le meme chunk peut
    apparaitre dans plusieurs requetes :
    - question originale ;
    - reformulation LLM ;
    - expansion synonymes.

    Principe de fusion :
    - on garde un seul exemplaire du chunk ;
    - on conserve son meilleur score FAISS ;
    - on ajoute un petit bonus s'il a ete retrouve par plusieurs requetes.

    Ce bonus reste volontairement faible : il signale la robustesse du match sans
    ecraser BM25, les ajustements de profil ou le reranker LLM.
    """
    meilleurs_scores: dict[int, float] = {}
    nombre_requetes_match: dict[int, int] = {}

    for _, resultats_requete in resultats_par_requete:
        indices_vus_pour_requete: set[int] = set()

        for score, indice in resultats_requete:
            if indice == -1 or indice < 0 or indice >= len(morceaux):
                continue

            morceau = morceaux[indice]

            if not filtre.accepte(morceau):
                continue

            score_float = float(score)
            meilleurs_scores[indice] = max(
                score_float,
                meilleurs_scores.get(indice, score_float),
            )

            if indice not in indices_vus_pour_requete:
                nombre_requetes_match[indice] = nombre_requetes_match.get(indice, 0) + 1
                indices_vus_pour_requete.add(indice)

    candidats: list[tuple[float, Morceau]] = []

    for indice, score in meilleurs_scores.items():
        bonus_multi_requetes = 0.03 * max(
            0,
            nombre_requetes_match.get(indice, 1) - 1,
        )
        candidats.append((score + bonus_multi_requetes, morceaux[indice]))

    return sorted(candidats, key=lambda candidat: candidat[0], reverse=True)


def diversifier_resultats_par_document(
    resultats: list[ResultatRecherche],
    top_k: int,
    max_par_document: int = 2,
) -> list[ResultatRecherche]:
    """
    Garde les meilleurs resultats, mais evite qu'un seul document monopolise tout.

    Utile pour les questions "toutes les solutions", ou l'objectif est de couvrir
    plusieurs produits plutot que d'extraire dix chunks d'une meme datasheet.
    """
    selection: list[ResultatRecherche] = []
    compte_par_document: dict[str, int] = {}

    for resultat in resultats:
        fichier = resultat.morceau.fichier

        if compte_par_document.get(fichier, 0) >= max_par_document:
            continue

        selection.append(resultat)
        compte_par_document[fichier] = compte_par_document.get(fichier, 0) + 1

        if len(selection) >= top_k:
            return selection

    for resultat in resultats:
        if resultat in selection:
            continue

        selection.append(resultat)

        if len(selection) >= top_k:
            break

    return selection


def contient_valeurs_techniques(texte: str) -> bool:
    """Repere les chunks qui contiennent probablement des valeurs techniques."""
    texte_minuscule = texte.lower()
    contient_nombre = bool(re.search(r"\d", texte_minuscule))
    contient_unite = any(
        unite in texte_minuscule
        for unite in [
            "watts",
            "watt",
            "tb",
            "tib",
            "pb",
            "pib",
            "gb/s",
            "iops",
            "latency",
            "physical",
            "technical specifications",
        ]
    )
    return contient_nombre and contient_unite


def filtrer_resultats_pour_valeurs_techniques(
    question: str,
    resultats: list[ResultatRecherche],
    question_demande_valeurs_techniques_fn: Callable[[str], bool] | None = None,
) -> list[ResultatRecherche]:
    """
    Pour les questions techniques, evite de noyer le modele avec du contenu
    marketing si des chunks de specifications sont disponibles.

    La detection de l'intention technique peut rester dans `rag.engine` et etre
    injectee ici. Sans fonction injectee, on applique seulement le filtre si la
    question contient directement des indices techniques simples.
    """
    if question_demande_valeurs_techniques_fn is None:
        question_est_technique = any(
            mot in question.lower()
            for mot in [
                "consommation",
                "consomamtion",
                "electrique",
                "\u00e9lectrique",
                "watts",
                "watt",
                "puissance",
                "capacity",
                "capacite",
                "capacit\u00e9",
            ]
        )
    else:
        question_est_technique = question_demande_valeurs_techniques_fn(question)

    if not question_est_technique:
        return resultats

    resultats_techniques = [
        resultat
        for resultat in resultats
        if contient_valeurs_techniques(resultat.morceau.texte)
    ]

    if not resultats_techniques:
        return resultats

    profil = profil_domaine_actif()
    termes_fichiers_focus = profil.technical_focus_filename_terms(question)

    if termes_fichiers_focus:
        resultats_focus = [
            resultat
            for resultat in resultats_techniques
            if any(
                terme in resultat.morceau.fichier.lower()
                for terme in termes_fichiers_focus
            )
        ]

        if resultats_focus:
            resultats_techniques = resultats_focus

    question_minuscule = question.lower()

    if any(
        mot in question_minuscule
        for mot in [
            "consommation",
            "consomamtion",
            "electrique",
            "\u00e9lectrique",
            "watts",
            "watt",
            "puissance",
        ]
    ):
        resultats_watts = [
            resultat
            for resultat in resultats_techniques
            if "watt" in resultat.morceau.texte.lower()
        ]

        if resultats_watts:
            return resultats_watts

    return resultats_techniques
