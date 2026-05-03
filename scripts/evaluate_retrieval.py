"""
Evaluation simple du retrieval RAG.

Ce script ne juge pas la beaute de la reponse finale. Il mesure une etape plus
fondamentale : est-ce que le RAG retrouve les bons documents et les bons signaux
avant d'appeler le LLM ?

Pourquoi c'est utile pour une demo SE ?
- On peut expliquer la qualite avec des chiffres, pas seulement une impression.
- On compare facilement FAISS + BM25 avec ou sans BGE.
- On detecte les regressions quand on change le chunking, top_k ou candidate_k.

Usage :

    OPENAI_API_KEY="..." .venv311/bin/python scripts/evaluate_retrieval.py
    .venv311/bin/python scripts/evaluate_retrieval.py --bge
    .venv311/bin/python scripts/evaluate_retrieval.py --top-k 8 --candidate-k 120
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_FILE = BASE_DIR / "data" / "evaluation" / "questions.json"

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import rag_pdf  # noqa: E402


def charger_questions(chemin: Path) -> list[dict]:
    """Charge les cas de test depuis un JSON volontairement simple."""
    with chemin.open("r", encoding="utf-8") as fichier:
        return json.load(fichier)


def evaluer_question(
    cas: dict,
    top_k: int,
    candidate_k: int,
    min_score: float,
    query_rewrite: bool,
    bge: bool,
) -> dict:
    """Execute le retrieval et calcule quelques scores faciles a expliquer."""
    debut = time.perf_counter()
    resultats = rag_pdf.rechercher(
        cas["question"],
        top_k=top_k,
        candidate_k=candidate_k,
        min_score=min_score,
        utiliser_query_rewrite_llm=query_rewrite,
        utiliser_reranker_llm=False,
        utiliser_cross_encoder_reranker=bge,
    )
    duree = time.perf_counter() - debut

    docs_retrouves = [resultat.morceau.fichier for resultat in resultats]
    docs_uniques = sorted(set(docs_retrouves))
    texte_retrouve = "\n".join(
        " ".join(
            [
                resultat.morceau.fichier,
                getattr(resultat.morceau, "titre", ""),
                resultat.morceau.texte,
            ]
        )
        for resultat in resultats
    ).lower()

    expected_docs = cas.get("expected_docs", [])
    expected_keywords = cas.get("expected_keywords", [])

    docs_hits = [doc for doc in expected_docs if doc in docs_uniques]
    keyword_hits = [
        keyword
        for keyword in expected_keywords
        if keyword.lower() in texte_retrouve
    ]

    doc_recall = len(docs_hits) / len(expected_docs) if expected_docs else 1.0
    keyword_recall = (
        len(keyword_hits) / len(expected_keywords) if expected_keywords else 1.0
    )

    return {
        "id": cas.get("id", cas["question"][:40]),
        "question": cas["question"],
        "duration_s": round(duree, 2),
        "bge_active": rag_pdf.dernier_reranker_bge_actif(),
        "retrieved_docs": docs_uniques,
        "expected_docs": expected_docs,
        "docs_hits": docs_hits,
        "doc_recall": round(doc_recall, 2),
        "expected_keywords": expected_keywords,
        "keyword_hits": keyword_hits,
        "keyword_recall": round(keyword_recall, 2),
        "top_sources": [
            {
                "rank": index,
                "doc": resultat.morceau.fichier,
                "page": resultat.morceau.page,
                "score": round(resultat.score_final, 3),
            }
            for index, resultat in enumerate(resultats, start=1)
        ],
    }


def afficher_tableau(resultats: list[dict]) -> None:
    """Affiche un resume lisible dans le terminal."""
    print("\nEvaluation retrieval")
    print("=" * 80)
    for resultat in resultats:
        print(
            f"{resultat['id']:<24} "
            f"docs={resultat['doc_recall']:.2f} "
            f"keywords={resultat['keyword_recall']:.2f} "
            f"bge={resultat['bge_active']} "
            f"time={resultat['duration_s']:.2f}s"
        )
        print("  docs hits:", ", ".join(resultat["docs_hits"]) or "-")
        print("  keywords:", ", ".join(resultat["keyword_hits"]) or "-")

    if resultats:
        moyenne_docs = sum(r["doc_recall"] for r in resultats) / len(resultats)
        moyenne_keywords = sum(r["keyword_recall"] for r in resultats) / len(resultats)
        print("-" * 80)
        print(f"Moyenne docs recall    : {moyenne_docs:.2f}")
        print(f"Moyenne keyword recall : {moyenne_keywords:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evalue le retrieval du RAG.")
    parser.add_argument("--file", type=Path, default=DEFAULT_EVAL_FILE)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--candidate-k", type=int, default=120)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--query-rewrite", action="store_true")
    parser.add_argument("--bge", action="store_true")
    parser.add_argument("--json", action="store_true", help="Sortie JSON detaillee.")
    args = parser.parse_args()

    questions = charger_questions(args.file)
    resultats = []
    for cas in questions:
        print(f"Evaluation: {cas.get('id', cas['question'][:40])}...", flush=True)
        resultat = evaluer_question(
            cas,
            top_k=args.top_k,
            candidate_k=args.candidate_k,
            min_score=args.min_score,
            query_rewrite=args.query_rewrite,
            bge=args.bge,
        )
        resultats.append(resultat)
        print(
            f"  docs={resultat['doc_recall']:.2f} "
            f"keywords={resultat['keyword_recall']:.2f} "
            f"time={resultat['duration_s']:.2f}s",
            flush=True,
        )

    if args.json:
        print(json.dumps(resultats, ensure_ascii=False, indent=2))
    else:
        afficher_tableau(resultats)


if __name__ == "__main__":
    main()
