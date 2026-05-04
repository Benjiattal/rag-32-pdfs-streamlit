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
| Retrieval lexical / BM25 / expansion / multi-requetes / fusion candidats / filtres simples | `rag/retrieval.py` | Fonctions locales extraites |
| Query rewrite LLM | `rag/query_rewrite.py` | Extrait |
| Profils metier optionnels | `rag/profiles.py`, `rag/domain_profiles/`, `rag/everpure.py` | Extrait |
| Search orchestration | `rag/search.py` | Extrait |
| Reranking LLM / BGE | `rag/reranking.py` | Extrait |
| Prompt et contexte long | `rag/prompting.py` | Extrait |
| Pipeline final / CLI | `rag/engine.py` | Reste orchestrateur |

## Pourquoi ne pas tout extraire d'un coup ?

Un RAG est un pipeline : ingestion, chunks, embeddings, retrieval, reranking,
prompt, generation. Ces etapes se passent des objets entre elles. Une extraction
trop rapide peut creer des imports circulaires ou changer subtilement le
comportement.

Exemple : les aides propres au profil Everpure, le prompt final, le reranking
BGE/LLM, la reformulation LLM et la recherche `rechercher()` ont ete sortis du
moteur principal. Le prochain risque concerne surtout `demander()` et la CLI,
car ces points restent la facade historique appelee par `rag_pdf.py` et
Streamlit.

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
    "rag/search.py",
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

La prochaine extraction logique est `rag/pipeline.py` ou `rag/cli.py`.

Elle devrait contenir :

- la fonction `demander()` ;
- la lecture des variables `TOP_K`, `CANDIDATE_K`, `MIN_SCORE` ;
- l'enchainement `rechercher()` -> `construire_prompt()` -> `appeler_modele()` ;
- eventuellement l'affichage CLI des sources.

La CLI peut rester dans `rag/engine.py` tant que l'API historique n'est pas
totalement stabilisee. C'est volontaire : mieux vaut une facade encore simple
qu'un decoupage trop fin difficile a expliquer en demo.
