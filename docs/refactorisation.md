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
| Profils metier optionnels | `rag/profiles.py`, `rag/domain_profiles/`, `rag/everpure.py` | Extrait |
| Retrieval orchestration | `rag/engine.py` | Reste orchestrateur |
| Reranking LLM / BGE | `rag/reranking.py` | Facade a extraire |
| Prompt et contexte long | `rag/prompting.py` | Extrait |

## Pourquoi ne pas tout extraire d'un coup ?

Un RAG est un pipeline : ingestion, chunks, embeddings, retrieval, reranking,
prompt, generation. Ces etapes se passent des objets entre elles. Une extraction
trop rapide peut creer des imports circulaires ou changer subtilement le
comportement.

Exemple : les aides propres au profil Everpure et le prompt final ont ete sortis
du moteur principal. Le prochain risque concerne surtout le reranking BGE/LLM :
il melange cache, worker local et appels optionnels, donc l'extraction doit
rester prudente.

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

La prochaine extraction logique est `rag/reranking.py`.

Elle devrait contenir :

- le cache du reranker LLM ;
- le prompt strict JSON du reranker LLM ;
- le chargement du CrossEncoder/BGE ;
- le worker BGE externe ;
- la fonction de reranking commune.

La fonction `rechercher()` peut rester dans `rag/engine.py` tant que
l'orchestration globale n'est pas stabilisee. C'est un choix volontaire : mieux
vaut un orchestrateur encore un peu large qu'un decoupage premature difficile a
expliquer.
