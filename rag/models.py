"""
Structures de donnees partagees du RAG.

Ces dataclasses definissent le contrat entre ingestion, retrieval, reranking,
generation et interface Streamlit.

Pourquoi les sortir de `engine.py` ?
Elles sont le vocabulaire commun du projet. Les garder dans un module leger
permet de les importer sans charger tout le moteur RAG.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Morceau:
    """Un morceau de texte extrait d'un PDF ou d'une page web."""

    texte: str
    fichier: str
    page: int
    numero: int
    source_type: str = "pdf"
    url: str = ""
    titre: str = ""
    date_consultation: str = ""


@dataclass
class ResultatRecherche:
    """Un resultat enrichi apres recherche FAISS et reranking."""

    morceau: Morceau
    score_semantique: float
    score_lexical: float
    score_final: float


@dataclass
class FiltreMetadata:
    """Filtres simples sur les metadonnees des morceaux."""

    fichiers: set[str] | None = None
    page_min: int | None = None
    page_max: int | None = None
    source_types: set[str] | None = None

    def accepte(self, morceau: Morceau) -> bool:
        """Retourne True si le morceau respecte les filtres demandes."""
        if self.source_types and getattr(morceau, "source_type", "pdf") not in self.source_types:
            return False

        if self.fichiers and morceau.fichier not in self.fichiers:
            return False

        if self.page_min is not None and morceau.page < self.page_min:
            return False

        if self.page_max is not None and morceau.page > self.page_max:
            return False

        return True


@dataclass
class ParametresRecherche:
    """
    Parametres reellement utilises par le retrieval.

    L'utilisateur choisit des valeurs dans l'interface, mais le RAG peut parfois
    elargir automatiquement la recherche. Cette classe garde la decision
    explicite, donc facile a afficher dans la trace RAG.
    """

    top_k: int
    candidate_k: int
    min_score: float
    raison: str
