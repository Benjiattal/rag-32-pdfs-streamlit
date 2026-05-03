"""
Chunking intelligent.

Responsabilite :
transformer du texte brut en morceaux courts, lisibles et contextualises.

Principe RAG :
la qualite de la recherche depend beaucoup de la qualite des chunks. On essaie
donc de respecter les phrases, les sections et les titres au lieu de couper
brutalement tous les N caracteres.
"""

from rag.engine import (  # noqa: F401
    decouper_texte,
    decouper_texte_par_sections,
    extraire_sections_simples,
    ligne_parasite_pdf,
    est_titre_section_probable,
    nettoyer_titre_section,
    decouper_en_phrases,
    decouper_phrase_trop_longue,
)
