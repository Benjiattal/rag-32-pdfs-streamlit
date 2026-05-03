"""
Chargement de l'environnement local.

Responsabilite :
charger le fichier `.env` uniquement quand une action en a besoin.

Pourquoi un module dedie ?
Plusieurs briques du RAG peuvent avoir besoin de la cle OpenAI ou de variables
locales. En isolant cette fonction, on evite les imports circulaires avec
`rag.engine`.
"""

from __future__ import annotations


def charger_env_local() -> None:
    """
    Charge le fichier `.env` uniquement quand c'est necessaire.

    Streamlit recharge souvent l'application. On evite donc d'importer
    `python-dotenv` au demarrage global, et on le charge seulement dans les
    chemins actifs : indexation, recherche CLI ou pipeline complet.
    """
    from dotenv import load_dotenv

    load_dotenv()
