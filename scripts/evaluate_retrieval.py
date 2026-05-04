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
    .venv311/bin/python scripts/evaluate_retrieval.py --output reports/retrieval.json
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
        questions = json.load(fichier)

    if not isinstance(questions, list):
        raise ValueError("Le fichier d'evaluation doit contenir une liste de cas.")

    for index, cas in enumerate(questions, start=1):
        if not isinstance(cas, dict):
            raise ValueError(f"Cas #{index}: chaque cas doit etre un objet JSON.")
        if not cas.get("question"):
            raise ValueError(f"Cas #{index}: le champ 'question' est obligatoire.")

    return questions


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


def calculer_resume(resultats: list[dict]) -> dict:
    """Calcule un resume agregé facile a lire ou a exporter."""
    if not resultats:
        return {
            "questions": 0,
            "mean_doc_recall": 0.0,
            "mean_keyword_recall": 0.0,
            "mean_duration_s": 0.0,
        }

    return {
        "questions": len(resultats),
        "mean_doc_recall": round(
            sum(r["doc_recall"] for r in resultats) / len(resultats),
            3,
        ),
        "mean_keyword_recall": round(
            sum(r["keyword_recall"] for r in resultats) / len(resultats),
            3,
        ),
        "mean_duration_s": round(
            sum(r["duration_s"] for r in resultats) / len(resultats),
            3,
        ),
    }


def afficher_tableau(resultats: list[dict], resume: dict) -> None:
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

    print("-" * 80)
    print(f"Questions              : {resume['questions']}")
    print(f"Moyenne docs recall    : {resume['mean_doc_recall']:.2f}")
    print(f"Moyenne keyword recall : {resume['mean_keyword_recall']:.2f}")
    print(f"Temps moyen retrieval  : {resume['mean_duration_s']:.2f}s")


def sauvegarder_rapport(chemin: Path, configuration: dict, resultats: list[dict], resume: dict) -> None:
    """Sauvegarde un rapport JSON pour comparer deux runs plus tard."""
    chemin.parent.mkdir(parents=True, exist_ok=True)
    rapport = {
        "configuration": configuration,
        "summary": resume,
        "results": resultats,
    }
    chemin.write_text(
        json.dumps(rapport, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def verifier_seuils(
    resume: dict,
    fail_under_doc_recall: float | None,
    fail_under_keyword_recall: float | None,
) -> bool:
    """
    Retourne True si les seuils sont respectes.

    Ces seuils sont optionnels. Ils deviennent utiles quand on veut utiliser le
    benchmark comme garde-fou apres une refactorisation ou un changement de
    chunking.
    """
    ok = True

    if (
        fail_under_doc_recall is not None
        and resume["mean_doc_recall"] < fail_under_doc_recall
    ):
        print(
            f"ECHEC: mean_doc_recall={resume['mean_doc_recall']:.2f} "
            f"< seuil {fail_under_doc_recall:.2f}",
            file=sys.stderr,
        )
        ok = False

    if (
        fail_under_keyword_recall is not None
        and resume["mean_keyword_recall"] < fail_under_keyword_recall
    ):
        print(
            f"ECHEC: mean_keyword_recall={resume['mean_keyword_recall']:.2f} "
            f"< seuil {fail_under_keyword_recall:.2f}",
            file=sys.stderr,
        )
        ok = False

    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Evalue le retrieval du RAG.")
    parser.add_argument("--file", type=Path, default=DEFAULT_EVAL_FILE)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--candidate-k", type=int, default=120)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--query-rewrite", action="store_true")
    parser.add_argument("--bge", action="store_true")
    parser.add_argument("--json", action="store_true", help="Sortie JSON detaillee.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Chemin optionnel pour sauvegarder le rapport JSON.",
    )
    parser.add_argument(
        "--fail-under-doc-recall",
        type=float,
        default=None,
        help="Retourne un code erreur si la moyenne doc_recall est sous ce seuil.",
    )
    parser.add_argument(
        "--fail-under-keyword-recall",
        type=float,
        default=None,
        help="Retourne un code erreur si la moyenne keyword_recall est sous ce seuil.",
    )
    args = parser.parse_args()

    configuration = {
        "file": str(args.file),
        "top_k": args.top_k,
        "candidate_k": args.candidate_k,
        "min_score": args.min_score,
        "query_rewrite": args.query_rewrite,
        "bge": args.bge,
    }

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

    resume = calculer_resume(resultats)

    if args.json:
        print(
            json.dumps(
                {
                    "configuration": configuration,
                    "summary": resume,
                    "results": resultats,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        afficher_tableau(resultats, resume)

    if args.output:
        chemin_sortie = args.output
        if not chemin_sortie.is_absolute():
            chemin_sortie = BASE_DIR / chemin_sortie
        sauvegarder_rapport(chemin_sortie, configuration, resultats, resume)
        print(f"\nRapport JSON sauvegarde : {chemin_sortie}")

    seuils_ok = verifier_seuils(
        resume,
        fail_under_doc_recall=args.fail_under_doc_recall,
        fail_under_keyword_recall=args.fail_under_keyword_recall,
    )

    if not seuils_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
