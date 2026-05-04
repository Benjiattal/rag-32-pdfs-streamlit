"""
Package principal du RAG documentaire.

Ce fichier reste volontairement leger.

Pourquoi ?
`rag.engine` importe `rag.config` et `rag.models`. Si `rag.__init__` importait
automatiquement `rag.engine`, alors le simple fait de faire `import rag.config`
declencherait tout le moteur et pourrait creer des imports circulaires.

La facade de compatibilite `rag_pdf.py` importe explicitement `rag.engine`.
"""

__all__: list[str] = []
