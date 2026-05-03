"""
Fonctions LLM du RAG.

Ce module porte les interactions OpenAI liees au langage :
- creation du client OpenAI ;
- appel du modele de generation finale.

Le prompt RAG reste encore dans `engine.py`, car il depend d'heuristiques de
retrieval qui seront extraites plus tard. Cette extraction partielle reduit deja
le cote monolithique sans deplacer trop de logique a la fois.
"""

from __future__ import annotations

import os

from rag.config import DEFAULT_CHAT_MODEL, DEFAULT_TEMPERATURE


def creer_client_openai():
    """
    Cree le client OpenAI.

    Le script ne met jamais la cle API dans le code.
    Il lit seulement la variable d'environnement OPENAI_API_KEY.

    Si tu utilises macOS Keychain, charge d'abord la cle avec :

    export OPENAI_API_KEY="$(security find-generic-password -a "$USER" -s "OPENAI_API_KEY" -w)"
    """
    from openai import OpenAI

    cle_api = os.getenv("OPENAI_API_KEY")

    if not cle_api or cle_api == "sk-...":
        raise ValueError(
            "OPENAI_API_KEY est manquante. Charge-la depuis macOS Keychain "
            "ou renseigne-la dans un fichier .env local non partage."
        )

    return OpenAI(api_key=cle_api)


def appeler_modele(prompt: str, modele: str | None = None) -> str:
    """
    Appelle le modele OpenAI qui redige la reponse finale.

    Important :
    cette fonction ne fait pas de retrieval. Elle reçoit déjà un prompt construit
    avec le contexte sourcé, puis demande au modèle de rédiger en français.
    """
    client = creer_client_openai()
    modele = modele or os.getenv("OPENAI_MODEL", DEFAULT_CHAT_MODEL)
    temperature = float(os.getenv("OPENAI_TEMPERATURE", str(DEFAULT_TEMPERATURE)))

    consigne_systeme = (
        "Tu réponds toujours en français. "
        "Les sources peuvent être en anglais, mais ta réponse finale doit être en français. "
        "Conserve les noms de produits en anglais si nécessaire."
    )

    payload = {
        "model": modele,
        "input": [
            {"role": "system", "content": consigne_systeme},
            {"role": "user", "content": prompt},
        ],
    }

    # Certains modèles OpenAI n'exposent pas exactement les mêmes options.
    # Pour un POC comparatif, on tente d'abord avec temperature=0 pour obtenir
    # des réponses stables ; si le modèle refuse ce paramètre, on relance sans.
    try:
        reponse = client.responses.create(**payload, temperature=temperature)
    except Exception as erreur:
        message = str(erreur).lower()
        if "temperature" not in message:
            raise
        reponse = client.responses.create(**payload)

    return reponse.output_text
