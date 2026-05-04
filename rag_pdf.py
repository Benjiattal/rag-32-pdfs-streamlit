"""
Facade de compatibilite pour l'application RAG.

Historique :
au debut du projet, tout le moteur RAG vivait dans ce fichier. C'etait pratique
pour apprendre, mais le fichier est devenu trop gros.

Nouvelle architecture :
le code est maintenant expose via le package `rag/` :
- `rag.ingestion` : lecture PDF/web et indexation ;
- `rag.chunking` : decoupage intelligent par phrases/sections ;
- `rag.embeddings` : embeddings OpenAI et cache ;
- `rag.retrieval` : FAISS, BM25, filtres metadata, top_k adaptatif ;
- `rag.query_rewrite` : reformulation LLM optionnelle pour le retrieval ;
- `rag.reranking` : reranking hybride et LLM ;
- `rag.prompting` : contexte long, prompt final et consignes de citation ;
- `rag.llm` : appel au modele de generation.

Pourquoi garder `rag_pdf.py` ?
Pour ne pas casser l'existant :
- Streamlit peut continuer a faire `import rag_pdf` ;
- la CLI `python rag_pdf.py indexer` fonctionne toujours ;
- les anciens scripts restent compatibles.
"""

from rag.engine import *  # noqa: F403
from rag.engine import main


if __name__ == "__main__":
    main()
