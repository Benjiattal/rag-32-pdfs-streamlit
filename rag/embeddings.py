"""
Embeddings OpenAI et cache local.

Responsabilite :
transformer les textes en vecteurs et eviter les appels OpenAI inutiles grace au
cache d'embeddings.
"""

from __future__ import annotations

import hashlib
import os
import pickle

from rag.config import (
    EMBEDDING_CACHE_FILE,
    INDEX_DIR,
    DEFAULT_EMBEDDING_MODEL,
)
from rag.llm import creer_client_openai


def cle_cache_embedding(texte: str, modele: str) -> str:
    """
    Calcule une cle stable pour le cache d'embeddings.

    Un embedding depend a la fois du texte ET du modele utilise. Si on change de
    modele d'embedding, on ne doit pas reutiliser les anciens vecteurs.
    """
    contenu = f"{modele}\0{texte}".encode("utf-8")
    return hashlib.sha256(contenu).hexdigest()


def charger_cache_embeddings() -> dict[str, list[float]]:
    """
    Charge le cache local des embeddings.

    Si le fichier n'existe pas ou s'il est illisible, on repart avec un cache
    vide. Cela evite qu'un cache abime bloque toute l'application.
    """
    if not EMBEDDING_CACHE_FILE.exists():
        return {}

    try:
        with EMBEDDING_CACHE_FILE.open("rb") as fichier:
            cache = pickle.load(fichier)
    except (OSError, EOFError, pickle.UnpicklingError):
        return {}

    if not isinstance(cache, dict):
        return {}

    return cache


def sauvegarder_cache_embeddings(cache: dict[str, list[float]]) -> None:
    """
    Sauvegarde le cache d'embeddings de maniere atomique.

    On ecrit d'abord dans un fichier temporaire, puis on remplace l'ancien cache
    avec os.replace(). Streamlit ne peut donc pas lire un fichier a moitie ecrit.
    """
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    fichier_temporaire = EMBEDDING_CACHE_FILE.with_suffix(".pkl.tmp")

    with fichier_temporaire.open("wb") as fichier:
        pickle.dump(cache, fichier)

    os.replace(fichier_temporaire, EMBEDDING_CACHE_FILE)


def statistiques_cache_embeddings() -> dict[str, int | bool]:
    """Retourne quelques infos simples sur le cache d'embeddings."""
    cache = charger_cache_embeddings()

    return {
        "cache_existe": EMBEDDING_CACHE_FILE.exists(),
        "nombre_embeddings": len(cache),
        "taille_octets": EMBEDDING_CACHE_FILE.stat().st_size
        if EMBEDDING_CACHE_FILE.exists()
        else 0,
    }


def creer_embeddings(textes: list[str]) -> "np.ndarray":
    """
    Transforme une liste de textes en vecteurs numeriques avec OpenAI.

    Un embedding est une representation mathematique du sens d'un texte.
    Deux textes proches dans le sens auront souvent des vecteurs proches.
    """
    try:
        import numpy as np
    except ModuleNotFoundError as erreur:
        raise ModuleNotFoundError(
            "La bibliotheque numpy manque. Lance d'abord : "
            "pip install -r requirements.txt"
        ) from erreur

    client = creer_client_openai()
    modele = os.getenv("OPENAI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    taille_lot = 100
    cache = charger_cache_embeddings()
    cache_modifie = False

    if not textes:
        return np.empty((0, 0), dtype=np.float32)

    cles = [cle_cache_embedding(texte, modele) for texte in textes]
    textes_a_creer: list[str] = []
    cles_a_creer: list[str] = []
    cles_deja_planifiees: set[str] = set()

    for cle, texte in zip(cles, textes):
        if cle in cache or cle in cles_deja_planifiees:
            continue

        textes_a_creer.append(texte)
        cles_a_creer.append(cle)
        cles_deja_planifiees.add(cle)

    for debut in range(0, len(textes_a_creer), taille_lot):
        lot = textes_a_creer[debut : debut + taille_lot]
        cles_lot = cles_a_creer[debut : debut + taille_lot]

        reponse = client.embeddings.create(
            model=modele,
            input=lot,
            encoding_format="float",
        )

        for cle, embedding in zip(cles_lot, reponse.data):
            cache[cle] = embedding.embedding
            cache_modifie = True

    if cache_modifie:
        sauvegarder_cache_embeddings(cache)

    tous_les_vecteurs = [cache[cle] for cle in cles]
    vecteurs = np.array(tous_les_vecteurs, dtype=np.float32)

    # Avec des vecteurs normalises, un produit scalaire devient equivalent a une
    # similarite cosinus. C'est le mode utilise par FAISS IndexFlatIP.
    normes = np.linalg.norm(vecteurs, axis=1, keepdims=True)
    normes[normes == 0] = 1
    vecteurs = vecteurs / normes

    return vecteurs
