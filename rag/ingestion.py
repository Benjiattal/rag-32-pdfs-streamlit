"""
Ingestion des sources.

Responsabilite :
lire les PDF, extraire les tableaux, recuperer les pages web autorisees et
construire l'index FAISS local.
"""

from rag.engine import (  # noqa: F401
    lister_pdfs,
    lister_sources_web,
    lister_documents_indexes,
    statistiques_index,
    lire_pdf,
    extraire_tableaux_page,
    convertir_tableau_en_markdown,
    extraire_extraits_techniques,
    lire_page_web,
    indexer_pages_web,
    charger_web_chunks,
    sauvegarder_web_chunks,
    indexer_pdf,
    charger_index,
    charger_morceaux_pickle,
)
