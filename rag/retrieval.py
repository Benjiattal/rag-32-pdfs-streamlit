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
import re
import unicodedata


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
    Ajoute des synonymes utiles uniquement pour la recherche.

    Important :
    - La question originale reste celle envoyee au modele.
    - Cette version enrichie sert seulement a mieux recuperer les chunks.

    Pourquoi ?
    Un utilisateur peut ecrire "GSI", alors que les PDF parlent de
    "Global System Integrator", "system integrator", "partners" ou "go-to-market".
    Sans expansion, FAISS peut passer a cote de passages pertinents.
    """
    enrichissements = []
    question_minuscule = question.lower()

    if "gsi" in question_minuscule:
        enrichissements.append(
            "global system integrator system integrator partner partners "
            "technology alliance program go-to-market business synergy "
            "joint solutions customer outcomes professional services "
            "managed services implementation deployment customer transformation"
        )

    if "evergreen" in question_minuscule:
        enrichissements.append(
            "evergreen architecture ever modern nondisruptive upgrade "
            "no data migration no planned downtime subscription"
        )

    if "everpure" in question_minuscule:
        enrichissements.append(
            "Everpure Platform Everpure Fusion Pure1 Evergreen Architecture "
            "Evergreen One Cloud Dedicated FlashArray FlashBlade unified data plane "
            "intelligent control plane policy driven automation AIOps"
        )

    if "solution" in question_minuscule or "solutions" in question_minuscule:
        enrichissements.append(
            "portfolio platform products services capabilities use cases "
            "FlashArray FlashBlade Pure1 Everpure Fusion Evergreen One "
            "Cloud Dedicated Portworx"
        )

    if any(
        mot in question_minuscule
        for mot in [
            "volumetrie",
            "volumetries",
            "volum\u00e9trie",
            "volum\u00e9tries",
            "capacite",
            "capacites",
            "capacit\u00e9",
            "capacit\u00e9s",
            "capacity",
            "capacities",
        ]
    ):
        enrichissements.append(
            "capacity capacities volumetry volume storage raw capacity effective capacity "
            "usable capacity maximum capacity up to PB PiB TB TiB data reduction "
            "FlashArray//XL FlashArray//ST FlashArray family FlashBlade//S "
            "FlashBlade//E FlashBlade//EXA Cloud Dedicated Cloud Block Store "
            "Portworx DirectFlash Shelf technical specifications"
        )

    mots_puissance = [
        "consommation",
        "consommations",
        "electrique",
        "electriques",
        "\u00e9lectrique",
        "\u00e9lectriques",
        "watts",
        "watt",
        "puissance",
        "power",
    ]
    if any(mot in question_minuscule for mot in mots_puissance):
        enrichissements.append(
            "power consumption watts watt typical peak physical specifications "
            "technical specifications physical capacity power and cooling "
            "2,635 3,455 2,475 3,160 2,115 2,700 566 667"
        )

    if "flasharray xl" in question_minuscule or "flasharray//xl" in question_minuscule:
        enrichissements.append(
            "FlashArray//XL FlashArray XL XL190 R5 XL170 R5 XL130 R5 "
            "DirectFlash Shelf technical specifications"
        )

    if "flasharray" in question_minuscule and any(
        mot in question_minuscule
        for mot in [
            "serveur",
            "serveurs",
            "server",
            "servers",
            "modele",
            "modeles",
            "mod\u00e8le",
            "mod\u00e8les",
            "gamme",
            "famille",
            "liste",
            "tous",
            "toutes",
        ]
    ):
        enrichissements.append(
            "FlashArray models FlashArray family product line storage arrays "
            "FlashArray//ST FlashArray//XL R5 FlashArray//X R5 FlashArray//C R5 "
            "FlashArray//E FlashArray Models Model Optimized For Strengths Capacity raw "
            "XL190 R5 XL170 R5 XL130 R5 X10 X20 X50 X70 X90 C20 C40 C60 C70 C90"
        )

    if "flashblade" in question_minuscule and any(
        mot in question_minuscule
        for mot in [
            "baie",
            "baies",
            "stockage",
            "gamme",
            "famille",
            "liste",
            "tous",
            "toutes",
            "modele",
            "modeles",
            "mod\u00e8le",
            "mod\u00e8les",
            "descriptif",
            "description",
        ]
    ):
        enrichissements.append(
            "FlashBlade product family storage arrays unified file object storage "
            "FlashBlade//S FlashBlade//S500 FlashBlade//E FlashBlade//EXA "
            "FlashBlade S FlashBlade E FlashBlade EXA AI HPC neocloud "
            "repository workloads unstructured data high performance scale-out "
            "capacity optimized disaggregated architecture metadata data nodes"
        )

    if not enrichissements:
        return question

    return question + "\n\nTermes de recherche additionnels : " + " ".join(enrichissements)
