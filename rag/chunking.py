"""
Chunking intelligent.

Responsabilite :
transformer du texte brut en morceaux courts, lisibles et contextualises.

Principe RAG :
la qualite de la recherche depend beaucoup de la qualite des chunks. On essaie
donc de respecter les phrases, les sections et les titres au lieu de couper
brutalement tous les N caracteres.
"""

from __future__ import annotations

import os
import re

from rag.config import (
    DEFAULT_CHUNK_SIZE_CHARS,
    DEFAULT_CHUNK_OVERLAP_SENTENCES,
)


def decouper_texte(
    texte: str,
    taille: int | None = None,
    chevauchement_phrases: int | None = None,
) -> list[str]:
    """
    Decoupe le texte en morceaux, aussi appeles "chunks", en respectant les phrases.

    taille = nombre approximatif de caracteres par morceau.
    chevauchement = nombre de phrases repetees entre deux morceaux voisins.
    """
    taille = taille or int(os.getenv("CHUNK_SIZE_CHARS", str(DEFAULT_CHUNK_SIZE_CHARS)))
    chevauchement_phrases = chevauchement_phrases or int(
        os.getenv("CHUNK_OVERLAP_SENTENCES", str(DEFAULT_CHUNK_OVERLAP_SENTENCES))
    )
    texte = " ".join(texte.split())

    if len(texte) <= taille:
        return [texte]

    phrases = decouper_en_phrases(texte)
    morceaux = []
    phrases_morceau: list[str] = []

    for phrase in phrases:
        phrase = phrase.strip()

        if not phrase:
            continue

        # Si une phrase est tres longue, on la coupe en morceaux de secours.
        # Cela arrive souvent avec des tableaux ou des PDF mal extraits.
        if len(phrase) > taille:
            if phrases_morceau:
                morceaux.append(" ".join(phrases_morceau))
                phrases_morceau = []

            morceaux.extend(decouper_phrase_trop_longue(phrase, taille))
            continue

        texte_candidat = " ".join(phrases_morceau + [phrase])

        if len(texte_candidat) <= taille:
            phrases_morceau.append(phrase)
            continue

        if phrases_morceau:
            morceaux.append(" ".join(phrases_morceau))

        # On garde les dernieres phrases du chunk precedent.
        # Cela preserve le contexte entre deux chunks voisins.
        chevauchement = phrases_morceau[-chevauchement_phrases:]
        phrases_morceau = chevauchement + [phrase]

        # Si le chevauchement rend le nouveau chunk trop grand, on le reduit.
        while len(" ".join(phrases_morceau)) > taille and len(phrases_morceau) > 1:
            phrases_morceau.pop(0)

    if phrases_morceau:
        morceaux.append(" ".join(phrases_morceau))

    return [morceau for morceau in morceaux if morceau.strip()]


def decouper_texte_par_sections(
    texte: str,
    taille: int | None = None,
    chevauchement_phrases: int | None = None,
) -> list[tuple[str, str]]:
    """
    Decoupe une page PDF en chunks en essayant de respecter les sections.

    Idee importante pour un RAG :
    deux phrases identiques ne veulent pas toujours dire la meme chose selon le
    titre qui les precede. Par exemple "capacity" peut parler d'une baie, d'un
    service cloud ou d'une option. On conserve donc le titre de section dans le
    texte indexe pour donner plus de contexte aux embeddings, a BM25 et au
    reranker.

    Cette fonction reste volontairement heuristique :
    - elle repere des lignes courtes qui ressemblent a des titres ;
    - elle decoupe ensuite chaque section par phrases ;
    - elle prefixe chaque chunk avec "TITRE DE SECTION".

    Pour un POC, c'est souvent le meilleur rapport qualite/simplicite.
    """
    taille = taille or int(os.getenv("CHUNK_SIZE_CHARS", str(DEFAULT_CHUNK_SIZE_CHARS)))
    chevauchement_phrases = chevauchement_phrases or int(
        os.getenv("CHUNK_OVERLAP_SENTENCES", str(DEFAULT_CHUNK_OVERLAP_SENTENCES))
    )

    sections = extraire_sections_simples(texte)
    morceaux: list[tuple[str, str]] = []

    for titre_section, texte_section in sections:
        titre_section = nettoyer_titre_section(titre_section)
        chunks_section = decouper_texte(
            texte_section,
            taille=taille,
            chevauchement_phrases=chevauchement_phrases,
        )

        for chunk in chunks_section:
            if titre_section:
                texte_chunk = f"TITRE DE SECTION: {titre_section}\n\n{chunk}"
            else:
                texte_chunk = chunk

            morceaux.append((texte_chunk, titre_section))

    return morceaux


def extraire_sections_simples(texte: str) -> list[tuple[str, str]]:
    """
    Regroupe les lignes d'une page par sections probables.

    Pourquoi "probables" ?
    Un PDF n'expose pas toujours une structure propre comme HTML. On doit donc
    inferer les titres a partir de signaux simples : ligne courte, isolee, peu de
    ponctuation, forme numerotee, majuscules, etc.
    """
    lignes_originales = [ligne.strip() for ligne in texte.splitlines()]
    sections: list[tuple[str, list[str]]] = []
    titre_courant = ""
    lignes_courantes: list[str] = []
    ligne_precedente_vide = True

    def pousser_section() -> None:
        if lignes_courantes:
            sections.append((titre_courant, lignes_courantes.copy()))
            lignes_courantes.clear()

    for ligne in lignes_originales:
        if not ligne:
            ligne_precedente_vide = True
            continue

        ligne_nettoyee = " ".join(ligne.split())

        if ligne_parasite_pdf(ligne_nettoyee):
            continue

        if est_titre_section_probable(ligne_nettoyee, ligne_precedente_vide):
            pousser_section()
            titre_courant = ligne_nettoyee
        else:
            lignes_courantes.append(ligne_nettoyee)

        ligne_precedente_vide = False

    pousser_section()

    if not sections:
        texte_nettoye = " ".join(texte.split())
        return [("", texte_nettoye)] if texte_nettoye else []

    return [
        (titre, " ".join(lignes))
        for titre, lignes in sections
        if " ".join(lignes).strip()
    ]


def ligne_parasite_pdf(ligne: str) -> bool:
    """
    Ignore les lignes de navigation ou d'habillage souvent presentes dans les PDF.

    Ces lignes ne portent pas de contenu metier. Si on les indexe comme titres,
    elles peuvent polluer la comparaison des sources et donner un contexte faux au
    chunk.
    """
    ligne_minuscule = ligne.lower()
    parasites = {
        "logo - pure storage",
        "logo - everpure",
        "table of contents",
        "contents",
    }

    if ligne_minuscule in parasites:
        return True

    if ligne_minuscule.startswith("\u00a9") or "all rights reserved" in ligne_minuscule:
        return True

    return False


def est_titre_section_probable(ligne: str, ligne_precedente_vide: bool) -> bool:
    """
    Detecte une ligne qui ressemble a un titre de section.

    On reste conservateur : une mauvaise detection de titre ajoute du bruit dans
    l'index. Les titres courts et isoles sont donc favorises.
    """
    if len(ligne) < 4 or len(ligne) > 90:
        return False

    if ligne.startswith(("|", "- ", "* ", "\u2022 ")):
        return False

    if re.search(r"\d\s*(?:TB|TiB|PB|PiB|GB/s|IOPS|W|watts?)\b", ligne, re.IGNORECASE):
        return False

    mots = ligne.split()
    if len(mots) > 12:
        return False

    caracteres_alpha = [caractere for caractere in ligne if caractere.isalpha()]
    if len(caracteres_alpha) < 3:
        return False

    se_termine_comme_phrase = ligne.endswith((".", "!", "?", ";", ":"))
    numerote = bool(re.match(r"^\d+(?:\.\d+)*\s+\S+", ligne))
    beaucoup_de_majuscules = sum(c.isupper() for c in caracteres_alpha) / len(caracteres_alpha) > 0.65

    if numerote:
        return True

    if beaucoup_de_majuscules and len(mots) <= 10:
        return True

    if ligne_precedente_vide and not se_termine_comme_phrase and len(mots) <= 9:
        return True

    return False


def nettoyer_titre_section(titre: str) -> str:
    """Nettoie un titre de section avant de l'ajouter au chunk."""
    titre = " ".join(titre.split())

    if len(titre) > 90:
        titre = titre[:90].rstrip()

    return titre


def decouper_en_phrases(texte: str) -> list[str]:
    """
    Decoupe un texte en phrases avec une regle simple.

    Ce n'est pas un parseur linguistique parfait, mais pour un petit RAG local,
    cela donne deja de meilleurs chunks qu'une coupe brute tous les N caracteres.
    """
    texte = " ".join(texte.split())
    return re.split(r"(?<=[.!?])\s+", texte)


def decouper_phrase_trop_longue(phrase: str, taille: int) -> list[str]:
    """
    Coupe une phrase trop longue en morceaux de secours.

    Principe : on essaie d'abord de respecter les phrases. Si un PDF produit une
    phrase geante, on accepte une coupe par caracteres pour ne pas bloquer.
    """
    morceaux = []

    for debut in range(0, len(phrase), taille):
        morceaux.append(phrase[debut : debut + taille])

    return morceaux
