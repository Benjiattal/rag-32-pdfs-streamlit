"""
Compare deux rapports JSON produits par `evaluate_retrieval.py`.

Objectif didactique :
- un benchmark seul donne une photo ;
- deux benchmarks compares montrent si une modification ameliore ou degrade
  le retrieval.

Usage :

    .venv311/bin/python scripts/compare_retrieval_reports.py \
      reports/retrieval-baseline.json \
      reports/retrieval-current.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


METRICS = {
    "mean_doc_recall": "doc_recall",
    "mean_keyword_recall": "keyword_recall",
    "mean_duration_s": "latency",
}


def charger_rapport(chemin: Path) -> dict:
    """Charge un rapport JSON et verifie sa structure minimale."""
    rapport = json.loads(chemin.read_text(encoding="utf-8"))
    if not isinstance(rapport, dict):
        raise ValueError(f"{chemin}: rapport JSON invalide.")
    if "summary" not in rapport or not isinstance(rapport["summary"], dict):
        raise ValueError(f"{chemin}: champ 'summary' manquant.")
    return rapport


def formater_delta(delta: float, suffixe: str = "") -> str:
    """Formate un ecart avec signe explicite pour lecture rapide."""
    signe = "+" if delta >= 0 else "-"
    return f"{signe}{abs(delta):.2f}{suffixe}"


def comparer_resumes(baseline: dict, current: dict) -> list[dict]:
    """Retourne les deltas entre deux sections summary."""
    lignes = []
    for cle, libelle in METRICS.items():
        valeur_base = float(baseline.get(cle, 0.0))
        valeur_courante = float(current.get(cle, 0.0))
        lignes.append(
            {
                "metric": libelle,
                "baseline": valeur_base,
                "current": valeur_courante,
                "delta": valeur_courante - valeur_base,
            }
        )
    return lignes


def afficher_comparaison(lignes: list[dict]) -> None:
    """Affiche une comparaison compacte dans le terminal."""
    print("\nComparaison retrieval")
    print("=" * 64)
    for ligne in lignes:
        suffixe = "s" if ligne["metric"] == "latency" else ""
        print(
            f"{ligne['metric']:<16} "
            f"{ligne['baseline']:.2f}{suffixe} -> "
            f"{ligne['current']:.2f}{suffixe}  "
            f"{formater_delta(ligne['delta'], suffixe)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare deux rapports JSON de benchmark retrieval."
    )
    parser.add_argument("baseline", type=Path)
    parser.add_argument("current", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    baseline = charger_rapport(args.baseline)
    current = charger_rapport(args.current)
    lignes = comparer_resumes(baseline["summary"], current["summary"])

    if args.json:
        print(json.dumps({"comparison": lignes}, ensure_ascii=False, indent=2))
    else:
        afficher_comparaison(lignes)


if __name__ == "__main__":
    main()
