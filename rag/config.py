"""
Configuration centrale du RAG.

Principe suivi :
un changement de chemin, de modele ou de valeur par defaut doit se faire ici,
pas au milieu du code de retrieval ou de l'interface.
"""

from __future__ import annotations

import os
from pathlib import Path


# Le SDK OpenAI utilise Pydantic. Sur certains environnements locaux, Pydantic
# peut scanner les entry points de tous les paquets installes pour charger des
# plugins. Ce scan n'apporte rien a ce POC et peut ralentir fortement le premier
# appel OpenAI. On le desactive donc explicitement avant tout import OpenAI.
os.environ.setdefault("PYDANTIC_DISABLE_PLUGINS", "1")


# `config.py` vit dans le package `rag/`.
# La racine fonctionnelle du projet reste le dossier parent, car les dossiers
# `data/`, `.streamlit/` et l'application Streamlit sont au niveau projet.
BASE_DIR = Path(__file__).resolve().parents[1]
PDF_DIR = Path(os.getenv("RAG_PDF_DIR", BASE_DIR / "data" / "pdfs")).expanduser()
INDEX_DIR = Path(os.getenv("RAG_INDEX_DIR", BASE_DIR / "data" / "index")).expanduser()
WEB_DIR = Path(os.getenv("RAG_WEB_DIR", BASE_DIR / "data" / "web")).expanduser()


# Fichiers crees par l'indexation.
FAISS_INDEX_FILE = INDEX_DIR / "index.faiss"
METADATA_FILE = INDEX_DIR / "morceaux.pkl"
WEB_CHUNKS_FILE = WEB_DIR / "web_chunks.pkl"
EMBEDDING_CACHE_FILE = INDEX_DIR / "embeddings_cache.pkl"
QUERY_REWRITE_CACHE_FILE = INDEX_DIR / "query_rewrite_cache.pkl"
LLM_RERANK_CACHE_FILE = INDEX_DIR / "llm_rerank_cache.pkl"


# Modeles OpenAI utilises par defaut.
DEFAULT_CHAT_MODEL = "gpt-4.1-nano"
DEFAULT_CHAT_MODELS = [
    "gpt-4.1-nano",
    "gpt-5.4-nano",
    "gpt-5.4-mini",
]
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_TEMPERATURE = 0


# Parametres retrieval.
DEFAULT_TOP_K = 8
DEFAULT_CANDIDATE_K = 80
DEFAULT_CONTEXT_TOKEN_BUDGET = 6000
DEFAULT_MIN_SCORE = 0.10
DEFAULT_CHUNK_SIZE_CHARS = 800
DEFAULT_CHUNK_MIN_CHARS = 500
DEFAULT_CHUNK_OVERLAP_SENTENCES = 2


# Web controle.
DEFAULT_ALLOWED_WEB_DOMAINS = "purestorage.com,docs.purestorage.com,support.purestorage.com"


# Options qualite, desactivees par defaut pour garder l'interface fluide.
DEFAULT_QUERY_REWRITE_LLM = False
DEFAULT_LLM_RERANKER = False
DEFAULT_LLM_RERANK_CANDIDATES = 12


# Reranker local BGE.
DEFAULT_CROSS_ENCODER_RERANKER = False
DEFAULT_CROSS_ENCODER_MODEL = "BAAI/bge-reranker-base"
DEFAULT_CROSS_ENCODER_RERANK_CANDIDATES = 20
DEFAULT_BGE_WORKER_TIMEOUT_SECONDS = 5


# Versions de prompt/cache : les changer invalide naturellement les caches
# associes sans supprimer les fichiers a la main.
QUERY_REWRITE_PROMPT_VERSION = "v2"
LLM_RERANK_PROMPT_VERSION = "v1"


# Poids du score hybride.
POIDS_SCORE_SEMANTIQUE = 0.70
POIDS_SCORE_BM25 = 0.30
