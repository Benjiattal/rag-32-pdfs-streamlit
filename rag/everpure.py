"""
Heuristiques optionnelles pour le profil Everpure / Pure Storage.

Important : ce module n'est pas le moteur RAG generique. Il contient uniquement
les ajustements metier actives quand `RAG_DOMAIN_PROFILE=everpure`.

Pourquoi separer ce code ?
- Le RAG doit rester reutilisable avec n'importe quel corpus PDF.
- Les termes comme FlashArray ou FlashBlade ne doivent pas influencer un autre domaine.
- On peut supprimer ou remplacer ce profil sans toucher au pipeline principal.
"""

from __future__ import annotations

import re

from rag.models import ResultatRecherche


def question_demande_inventaire_produit(question: str) -> bool:
    """Detecte une question d'inventaire produit pour le profil Everpure."""
    return question_demande_inventaire_flasharray(
        question
    ) or question_demande_inventaire_flashblade(question)


def prioriser_inventaires_produits(
    question: str,
    resultats: list[ResultatRecherche],
) -> list[ResultatRecherche]:
    """
    Applique les petits reclassements propres au profil Everpure.

    Le moteur generique appelle cette fonction seulement si le profil Everpure
    est actif. Les noms de produits restent ainsi confines dans ce module.
    """
    resultats = prioriser_inventaire_flasharray(question, resultats)
    return prioriser_inventaire_flashblade(question, resultats)


def construire_aides_prompt(
    question: str,
    resultats: list[ResultatRecherche],
) -> str:
    """
    Construit les aides d'extraction optionnelles du profil Everpure.

    Ces aides ne remplacent jamais le contexte : elles servent de checklist au
    LLM quand la question demande explicitement une liste de gammes ou modeles.
    """
    aides = [
        construire_aide_inventaire_flasharray(question, resultats),
        construire_aide_inventaire_flashblade(question, resultats),
    ]
    return "\n".join(aide for aide in aides if aide)


def nombre_modeles_flasharray(texte: str) -> int:
    """
    Compte combien de modèles FlashArray distincts apparaissent dans un chunk.

    Pour une question de type "liste tous les modèles", un chunk qui contient
    plusieurs modèles dans un tableau est souvent plus utile qu'un chunk narratif
    qui ne parle que d'un seul produit.
    """
    texte_minuscule = texte.lower()
    modeles = [
        "flasharray//st",
        "flasharray//xl",
        "flasharray//x",
        "flasharray//c",
        "flasharray//e",
        "//xl190",
        "//xl170",
        "//xl130",
        "xl190 r5",
        "xl170 r5",
        "xl130 r5",
        "x10",
        "x20",
        "x50",
        "x70",
        "x90",
        "c20",
        "c40",
        "c60",
        "c70",
        "c90",
    ]
    return sum(1 for modele in modeles if modele in texte_minuscule)


def prioriser_inventaire_flasharray(
    question: str,
    resultats: list[ResultatRecherche],
) -> list[ResultatRecherche]:
    """
    Reclasse les résultats pour les questions "liste/gamme FlashArray".

    Principe qualité :
    le score FAISS + BM25 reste la base, mais on ajoute un petit bonus aux chunks
    qui ressemblent à un tableau de modèles. Cela évite qu'un document SQL Server
    soit préféré seulement parce que la question contient le mot "serveur".
    """
    if not question_demande_inventaire_flasharray(question):
        return resultats

    resultats_ajustes: list[ResultatRecherche] = []

    for resultat in resultats:
        texte = resultat.morceau.texte
        fichier = resultat.morceau.fichier.lower()
        texte_minuscule = texte.lower()
        bonus = 0.0

        nb_modeles = nombre_modeles_flasharray(texte)
        if nb_modeles >= 4:
            bonus += 0.35
        elif nb_modeles >= 2:
            bonus += 0.18

        if "flasharray-family" in fichier:
            bonus += 0.20

        if "flasharray models" in texte_minuscule or "model optimized for" in texte_minuscule:
            bonus += 0.25

        # Les documents SQL Server restent utilisables, mais ils ne doivent pas
        # dominer une question qui demande la gamme FlashArray.
        if "sql-server" in fichier:
            bonus -= 0.12

        resultats_ajustes.append(
            ResultatRecherche(
                morceau=resultat.morceau,
                score_semantique=resultat.score_semantique,
                score_lexical=resultat.score_lexical,
                score_final=resultat.score_final + bonus,
            )
        )

    return sorted(resultats_ajustes, key=lambda resultat: resultat.score_final, reverse=True)


def question_demande_inventaire_flashblade(question: str) -> bool:
    """
    Detecte les questions qui demandent la liste de la gamme FlashBlade.

    Le but est d'éviter une réponse limitée à FlashBlade//S ou //E quand le
    corpus contient aussi FlashBlade//EXA.
    """
    question_minuscule = question.lower()

    if "flashblade" not in question_minuscule:
        return False

    mots_inventaire = [
        "liste",
        "lister",
        "tous",
        "toutes",
        "baie",
        "baies",
        "gamme",
        "famille",
        "modele",
        "modeles",
        "modèle",
        "modèles",
        "descriptif",
        "description",
        "stockage",
    ]

    return any(mot in question_minuscule for mot in mots_inventaire)


def nombre_modeles_flashblade(texte: str) -> int:
    """Compte les familles FlashBlade distinctes visibles dans un chunk."""
    texte_minuscule = texte.lower()
    modeles = [
        "flashblade//s",
        "flashblade//s500",
        "flashblade//e",
        "flashblade//exa",
        "flashblade exa",
    ]
    return sum(1 for modele in modeles if modele in texte_minuscule)


def prioriser_inventaire_flashblade(
    question: str,
    resultats: list[ResultatRecherche],
) -> list[ResultatRecherche]:
    """
    Reclasse les résultats pour les questions "liste/gamme FlashBlade".

    Principe :
    - on favorise les chunks qui mentionnent explicitement une famille FlashBlade ;
    - on donne un bonus supplémentaire à FlashBlade//EXA, car il est souvent dans
      un technical brief ou un blog et se fait masquer par les documents S500 ;
    - on réduit légèrement le poids des reference designs NVIDIA quand ils
      parlent surtout d'une configuration et pas de la gamme produit.
    """
    if not question_demande_inventaire_flashblade(question):
        return resultats

    resultats_ajustes: list[ResultatRecherche] = []

    for resultat in resultats:
        texte = resultat.morceau.texte
        texte_minuscule = texte.lower()
        fichier = resultat.morceau.fichier.lower()
        bonus = 0.0

        nb_modeles = nombre_modeles_flashblade(texte)
        if nb_modeles >= 3:
            bonus += 0.35
        elif nb_modeles >= 2:
            bonus += 0.22
        elif nb_modeles == 1:
            bonus += 0.10

        if "flashblade-exa" in fichier or "flashblade//exa" in texte_minuscule or "flashblade exa" in texte_minuscule:
            bonus += 0.42

        if "flashblade-e" in fichier or "flashblade//e" in texte_minuscule:
            bonus += 0.22

        if "flashblade-s" in fichier or "flashblade//s" in texte_minuscule:
            bonus += 0.18

        if "dgx" in fichier or "superpod" in fichier or "reference-design" in fichier:
            bonus -= 0.10

        resultats_ajustes.append(
            ResultatRecherche(
                morceau=resultat.morceau,
                score_semantique=resultat.score_semantique,
                score_lexical=resultat.score_lexical,
                score_final=resultat.score_final + bonus,
            )
        )

    return sorted(resultats_ajustes, key=lambda resultat: resultat.score_final, reverse=True)


def question_demande_inventaire_flasharray(question: str) -> bool:
    """
    Detecte les questions qui demandent la liste/gamme des modèles FlashArray.

    Exemple utilisateur : "liste de tous les serveurs flasharray".

    Remarque importante :
    dans le vocabulaire Pure/Everpure, on parle plutôt de "baies", "arrays" ou
    "modèles" FlashArray, pas vraiment de serveurs. Mais beaucoup d'utilisateurs
    emploient "serveur" comme terme générique pour désigner une appliance.
    Cette fonction traduit donc cette intention vers "liste des modèles".
    """
    question_minuscule = question.lower()

    if "flasharray" not in question_minuscule:
        return False

    mots_inventaire = [
        "liste",
        "lister",
        "tous",
        "toutes",
        "serveur",
        "serveurs",
        "server",
        "servers",
        "modele",
        "modeles",
        "modèle",
        "modèles",
        "gamme",
        "famille",
        "portfolio",
        "portefeuille",
    ]

    return any(mot in question_minuscule for mot in mots_inventaire)


def extraire_modeles_flasharray_detectes(
    resultats: list[ResultatRecherche],
) -> dict[str, list[int]]:
    """
    Extrait les modèles FlashArray explicitement visibles dans les chunks.

    Pourquoi ajouter cette étape ?
    Un LLM peut parfois résumer une famille produit et oublier des déclinaisons
    présentes dans le contexte. Pour une question "liste tous les modèles", on
    préfère l'aider avec une extraction simple et traçable.

    La valeur du dictionnaire contient les numéros de sources où le modèle a été
    vu. Ces numéros correspondent aux "Source N" du contexte.
    """
    patrons = {
        "FlashArray//ST": r"FlashArray//ST|//ST R5",
        "FlashArray//XL R5": r"FlashArray//XL R5|FlashArray//XL™|FlashArray//XL",
        "FlashArray//XL190 R5": r"FlashArray//XL190 R5|//XL190 R5|XL190 R5",
        "FlashArray//XL170 R5": r"FlashArray//XL170 R5|//XL170 R5|XL170 R5",
        "FlashArray//XL130 R5": r"FlashArray//XL130 R5|//XL130 R5|XL130 R5",
        "FlashArray//X R5": r"FlashArray//X R5|FlashArray//X™|FlashArray//X(?!L)",
        "FlashArray//C R5": r"FlashArray//C R5|FlashArray//C™|FlashArray//C",
        "FlashArray//E": r"FlashArray//E™|FlashArray//E",
        "FlashArray//X10": r"FlashArray//X10|//X10|X10",
        "FlashArray//X20": r"FlashArray//X20|//X20|X20",
        "FlashArray//X50": r"FlashArray//X50|//X50|X50",
        "FlashArray//X70": r"FlashArray//X70|//X70|X70",
        "FlashArray//X90": r"FlashArray//X90|//X90|X90",
        "FlashArray//C20": r"FlashArray//C20|//C20|C20",
        "FlashArray//C40": r"FlashArray//C40|//C40|C40",
        "FlashArray//C60": r"FlashArray//C60|//C60|C60",
        "FlashArray//C70": r"FlashArray//C70|//C70|C70",
        "FlashArray//C90": r"FlashArray//C90|//C90|C90",
    }
    modeles: dict[str, list[int]] = {}

    for numero_source, resultat in enumerate(resultats, start=1):
        texte = resultat.morceau.texte

        for modele, patron in patrons.items():
            if re.search(patron, texte, flags=re.IGNORECASE):
                modeles.setdefault(modele, []).append(numero_source)

    return modeles


def construire_aide_inventaire_flasharray(
    question: str,
    resultats: list[ResultatRecherche],
) -> str:
    """
    Produit une aide compacte pour les questions de liste FlashArray.

    Cette aide ne remplace pas le contexte : elle résume seulement les modèles
    détectés automatiquement dans les chunks déjà récupérés.
    """
    if not question_demande_inventaire_flasharray(question):
        return ""

    modeles = extraire_modeles_flasharray_detectes(resultats)

    if not modeles:
        return ""

    lignes = [
        "Aide d'extraction pour la question FlashArray :",
        "Les modèles suivants apparaissent explicitement dans les sources récupérées.",
        "Utilise cette liste pour éviter d'oublier une famille ou une déclinaison.",
    ]

    for modele, sources in modeles.items():
        sources_uniques = sorted(set(sources))
        refs = ", ".join(f"[{source}]" for source in sources_uniques[:4])
        lignes.append(f"- {modele} : {refs}")

    return "\n".join(lignes)


def extraire_modeles_flashblade_detectes(
    resultats: list[ResultatRecherche],
) -> dict[str, list[int]]:
    """
    Extrait les familles FlashBlade visibles dans les chunks récupérés.

    Cette extraction sert de checklist pour éviter que le modèle oublie EXA
    quand les sources récupérées contiennent surtout des documents S500.
    """
    patrons = {
        "FlashBlade//S": r"FlashBlade//S(?!500)|FlashBlade//S™",
        "FlashBlade//S500": r"FlashBlade//S500|S500",
        "FlashBlade//E": r"FlashBlade//E|FlashBlade//E™",
        "FlashBlade//EXA": r"FlashBlade//EXA|FlashBlade EXA|EXA",
    }
    modeles: dict[str, list[int]] = {}

    for numero_source, resultat in enumerate(resultats, start=1):
        texte = resultat.morceau.texte

        for modele, patron in patrons.items():
            if re.search(patron, texte, flags=re.IGNORECASE):
                modeles.setdefault(modele, []).append(numero_source)

    return modeles


def construire_aide_inventaire_flashblade(
    question: str,
    resultats: list[ResultatRecherche],
) -> str:
    """Construit une aide compacte pour les questions de gamme FlashBlade."""
    if not question_demande_inventaire_flashblade(question):
        return ""

    modeles = extraire_modeles_flashblade_detectes(resultats)

    if not modeles:
        return ""

    lignes = [
        "Aide d'extraction pour la question FlashBlade :",
        "Les familles suivantes apparaissent explicitement dans les sources récupérées.",
        "Utilise cette liste comme checklist, notamment pour ne pas oublier FlashBlade//EXA.",
    ]

    for modele, sources in modeles.items():
        sources_uniques = sorted(set(sources))
        refs = ", ".join(f"[{source}]" for source in sources_uniques[:4])
        lignes.append(f"- {modele} : {refs}")

    return "\n".join(lignes)
