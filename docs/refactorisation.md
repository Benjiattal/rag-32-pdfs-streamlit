# Refactorisation progressive

Ce document sert de garde-fou pour simplifier le code sans regression.

## Objectif

Le projet est volontairement didactique, mais `rag/engine.py` est devenu trop
large. La strategie retenue est donc progressive :

1. extraire les blocs les plus autonomes ;
2. conserver les memes noms de fonctions publics ;
3. verifier la syntaxe et quelques comportements simples apres chaque extraction ;
4. eviter de melanger refactorisation et changement fonctionnel.

## Etat actuel

| Domaine | Module cible | Etat |
| --- | --- | --- |
| Configuration | `rag/config.py` | Extrait |
| Modeles de donnees | `rag/models.py` | Extrait |
| Appels LLM | `rag/llm.py` | Extrait |
| Embeddings et cache | `rag/embeddings.py` | Extrait |
| Chunking | `rag/chunking.py` | Extrait |
| Environnement local | `rag/env.py` | Extrait |
| Ingestion PDF / Web | `rag/ingestion.py` | Extrait |
| Retrieval lexical / BM25 / expansion / multi-requetes / fusion candidats | `rag/retrieval.py` | Fonctions locales extraites |
| Retrieval orchestration / filtres | `rag/engine.py` puis `rag/retrieval.py` | A extraire prudemment |
| Reranking LLM / BGE | `rag/reranking.py` | Facade a extraire |
| Prompt et contexte long | futur `rag/prompting.py` | A faire apres retrieval |

## Pourquoi ne pas tout extraire d'un coup ?

Un RAG est un pipeline : ingestion, chunks, embeddings, retrieval, reranking,
prompt, generation. Ces etapes se passent des objets entre elles. Une extraction
trop rapide peut creer des imports circulaires ou changer subtilement le
comportement.

Exemple : le prompt utilise aujourd'hui des aides metier FlashArray/FlashBlade
qui dependent de la detection du type de question. Cette detection sert aussi au
retrieval adaptatif. Il faut donc isoler proprement cette logique avant de sortir
le prompt dans son propre module.

## Regles de verification

Apres chaque extraction :

```bash
.venv311/bin/python -B - <<'PY'
from pathlib import Path
for path in [
    "rag/config.py",
    "rag/models.py",
    "rag/llm.py",
    "rag/embeddings.py",
    "rag/chunking.py",
    "rag/env.py",
    "rag/ingestion.py",
    "rag/engine.py",
    "rag_pdf.py",
]:
    compile(Path(path).read_text(encoding="utf-8"), path, "exec")
    print(path, "syntax ok")
PY
```

Puis, si l'index et la cle OpenAI sont disponibles, lancer une question courte
depuis l'interface ou la CLI.

## Prochaine extraction recommandee

La prochaine extraction logique est `rag/retrieval.py`.

Elle doit etre faite avec plus de prudence que l'ingestion, car elle porte :

- la normalisation lexicale ;
- BM25 ;
- l'expansion de requete ;
- les filtres metadata ;
- les heuristiques FlashArray / FlashBlade ;
- les parametres adaptatifs.

Pour limiter le risque, il faudra probablement extraire d'abord les fonctions
pures de scoring et de normalisation, puis seulement ensuite la fonction
`rechercher()`.
