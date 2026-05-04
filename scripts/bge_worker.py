"""
Worker local pour le reranker BGE.

Pourquoi un worker séparé ?
Streamlit + FAISS + PyTorch peuvent se gêner sur macOS à cause des runtimes
natifs chargés dans le même processus. Pour garder l'interface stable, on lance
donc BGE dans un processus Python séparé :

1. Streamlit garde le pipeline rapide FAISS + BM25.
2. Ce worker charge `sentence-transformers` et le modèle BGE.
3. Streamlit envoie une question + des chunks en JSON.
4. Le worker renvoie les scores et l'ordre reranké.

Le protocole est volontairement simple : une requête JSON par ligne sur stdin,
une réponse JSON par ligne sur stdout.
"""

from __future__ import annotations

import json
import os
import sys


def main() -> None:
    """Charge le modèle une seule fois puis traite les requêtes en boucle."""
    modele_nom = os.getenv("RAG_CROSS_ENCODER_MODEL", "BAAI/bge-reranker-base")

    # Import lourd : il reste ici, hors du processus Streamlit.
    from sentence_transformers import CrossEncoder

    modele = CrossEncoder(modele_nom)
    print(json.dumps({"status": "ready", "model": modele_nom}), flush=True)

    for ligne in sys.stdin:
        ligne = ligne.strip()
        if not ligne:
            continue

        try:
            requete = json.loads(ligne)
            question = requete["question"]
            candidats = requete["candidates"]
            paires = [
                (question, " ".join(candidat["text"].split())[:1800])
                for candidat in candidats
            ]
            scores = modele.predict(paires)
            reponse = {
                "status": "ok",
                "scores": [float(score) for score in scores],
            }
        except Exception as erreur:  # pragma: no cover - garde-fou runtime
            reponse = {
                "status": "error",
                "error": f"{type(erreur).__name__}: {erreur}",
            }

        print(json.dumps(reponse), flush=True)


if __name__ == "__main__":
    main()
