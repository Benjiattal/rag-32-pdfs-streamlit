# Verification des performances

Ce document distingue deux types de performance :

1. la performance locale, sans appel reseau ;
2. la performance RAG complete, qui depend d'OpenAI, de FAISS, du cache et du
   reranker choisi.

## Mesure locale sans appel OpenAI

Commande utilisee :

```bash
.venv311/bin/python -B - <<'PY'
import time
for mod in [
    "rag.config",
    "rag.models",
    "rag.chunking",
    "rag.retrieval",
    "rag.llm",
    "rag.embeddings",
    "rag.engine",
    "rag.ingestion",
    "rag_pdf",
]:
    t = time.perf_counter()
    __import__(mod)
    print(f"{mod}: {time.perf_counter() - t:.3f}s")
PY
```

Resultat observe apres extraction du chunking :

| Module | Temps |
| --- | ---: |
| `rag.config` | 0.013s |
| `rag.models` | 0.013s |
| `rag.chunking` | 0.001s |
| `rag.retrieval` | environ 0.03s |
| `rag.llm` | 0.000s |
| `rag.embeddings` | 0.023s |
| `rag.ingestion` | 0.044s |
| `rag.engine` | environ 0.046s |
| `rag_pdf` | 0.000s |

Lecture :
les imports restent rapides. La refactorisation n'a pas ajoute de dependance
lourde au demarrage.

## Mesure du chunking

Commande utilisee :

```bash
.venv311/bin/python -B - <<'PY'
from time import perf_counter
from rag.chunking import decouper_texte_par_sections

texte = "\n".join([
    "FlashBlade//EXA",
    "",
    "FlashBlade//EXA is designed for AI and HPC environments. "
    "It separates metadata and data nodes. "
    "It supports large-scale unstructured data workloads.",
    "FlashBlade//S",
    "",
    "FlashBlade//S is a unified file and object storage platform. "
    "It is used for repositories, analytics and modern applications.",
] * 100)

t = perf_counter()
chunks = decouper_texte_par_sections(texte, taille=800, chevauchement_phrases=2)
print("chunking:", len(chunks), "chunks in", f"{perf_counter() - t:.3f}s")
PY
```

Resultat observe :

```text
chunking: 50 chunks in 0.002s
```

## Mesure RAG complete

Pour evaluer la qualite du retrieval :

```bash
.venv311/bin/python scripts/evaluate_retrieval.py
```

Cette mesure peut varier selon :

- l'etat du cache embeddings ;
- la presence de l'index FAISS ;
- le reranker active ou non ;
- la latence OpenAI ;
- le premier chargement du modele BGE.

## Points de vigilance

- BGE est souvent lent au premier appel, puis plus rapide grace au cache memoire.
- Le reranker LLM est plus variable, car il appelle un modele distant.
- Le cache embeddings evite de recalculer les vecteurs de textes deja vus.
- L'index FAISS, les PDF et les caches sont volontairement hors Git : ils sont
  reconstruits localement.
