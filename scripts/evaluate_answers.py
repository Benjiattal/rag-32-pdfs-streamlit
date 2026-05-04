"""
Evaluation légère de la réponse finale.

Ce script complete `evaluate_retrieval.py`.

Retrieval evaluation :
    Est-ce que les bons chunks remontent ?

Answer evaluation :
    Est-ce que la réponse finale contient les faits attendus, cite ses sources
    et évite d'inventer ?

Important :
Cette evaluation reste volontairement simple. Elle ne remplace pas un juge LLM
ou une validation humaine, mais elle donne un premier garde-fou reproductible.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_FILE = BASE_DIR / "data" / "evaluation" / "answers.json"

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import rag_pdf  # noqa: E402


def charger_cas(chemin: Path) -> list[dict]:
    """Charge les cas d'evaluation de réponse finale."""
    with chemin.open("r", encoding="utf-8") as fichier:
        cas = json.load(fichier)

    if not isinstance(cas, list):
        raise ValueError("Le fichier doit contenir une liste de cas.")

    for index, item in enumerate(cas, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Cas #{index}: objet JSON attendu.")
        if not item.get("question"):
            raise ValueError(f"Cas #{index}: champ 'question' obligatoire.")

    return cas


def contient_citation(reponse: str) -> bool:
    """Detecte une citation source au format [1], [2], etc."""
    return bool(re.search(r"\[\d+\]", reponse))


def reponse_dit_ne_pas_savoir(reponse: str) -> bool:
    """Detecte un refus d'inventer, en français ou en anglais."""
    texte = reponse.lower()
    signaux = [
        "je ne sais pas",
        "je n'ai pas",
        "not in the context",
        "i don't know",
        "i do not know",
        "the context does not",
        "is not provided",
    ]
    return any(signal in texte for signal in signaux)


def evaluer_reponse_textuelle(cas: dict, reponse: str) -> dict:
    """
    Evalue une réponse déjà générée.

    Les règles sont volontairement transparentes :
    - faits attendus présents ;
    - termes interdits absents ;
    - citations présentes si demandées ;
    - "je ne sais pas" présent si la question est hors corpus.
    """
    texte = reponse.lower()
    faits_attendus = cas.get("expected_facts", [])
    termes_interdits = cas.get("forbidden_terms", [])
    citations_requises = bool(cas.get("must_have_citation", False))
    inconnu_attendu = bool(cas.get("expected_unknown", False))

    faits_presents = [
        fait for fait in faits_attendus if fait.lower() in texte
    ]
    interdits_presents = [
        terme for terme in termes_interdits if terme.lower() in texte
    ]
    citation_ok = contient_citation(reponse) if citations_requises else True
    unknown_ok = reponse_dit_ne_pas_savoir(reponse) if inconnu_attendu else True

    facts_recall = (
        len(faits_presents) / len(faits_attendus)
        if faits_attendus
        else 1.0
    )
    hallucination_ok = not interdits_presents
    ok = facts_recall == 1.0 and hallucination_ok and citation_ok and unknown_ok

    return {
        "id": cas.get("id", cas["question"][:40]),
        "question": cas["question"],
        "ok": ok,
        "facts_recall": round(facts_recall, 2),
        "expected_facts": faits_attendus,
        "facts_present": faits_presents,
        "forbidden_terms_present": interdits_presents,
        "citation_ok": citation_ok,
        "unknown_ok": unknown_ok,
    }


def calculer_resume(resultats: list[dict]) -> dict:
    """Agrege les resultats de réponse finale."""
    if not resultats:
        return {
            "questions": 0,
            "success_rate": 0.0,
            "mean_facts_recall": 0.0,
        }

    return {
        "questions": len(resultats),
        "success_rate": round(
            sum(1 for resultat in resultats if resultat["ok"]) / len(resultats),
            3,
        ),
        "mean_facts_recall": round(
            sum(resultat["facts_recall"] for resultat in resultats) / len(resultats),
            3,
        ),
    }


def afficher_resultats(resultats: list[dict], resume: dict) -> None:
    """Affiche un resume terminal lisible."""
    print("\nEvaluation réponses finales")
    print("=" * 80)
    for resultat in resultats:
        statut = "OK" if resultat["ok"] else "FAIL"
        print(
            f"{resultat['id']:<28} {statut:<5} "
            f"facts={resultat['facts_recall']:.2f} "
            f"citation={resultat['citation_ok']} "
            f"unknown={resultat['unknown_ok']}"
        )
        if resultat["forbidden_terms_present"]:
            print("  forbidden:", ", ".join(resultat["forbidden_terms_present"]))

    print("-" * 80)
    print(f"Questions          : {resume['questions']}")
    print(f"Success rate       : {resume['success_rate']:.2f}")
    print(f"Mean facts recall  : {resume['mean_facts_recall']:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evalue les réponses finales du RAG.")
    parser.add_argument("--file", type=Path, default=DEFAULT_EVAL_FILE)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--fail-under-success-rate",
        type=float,
        default=None,
        help="Retourne un code erreur si le taux de succès est sous ce seuil.",
    )
    parser.add_argument(
        "--answer-file",
        type=Path,
        default=None,
        help=(
            "Fichier JSON optionnel contenant des réponses déjà générées. "
            "Format: {\"id\": \"réponse\"}."
        ),
    )
    args = parser.parse_args()

    cas_evaluation = charger_cas(args.file)
    reponses_fournies = {}
    if args.answer_file:
        reponses_fournies = json.loads(args.answer_file.read_text(encoding="utf-8"))

    resultats = []
    for cas in cas_evaluation:
        identifiant = cas.get("id", cas["question"][:40])
        if identifiant in reponses_fournies:
            reponse = reponses_fournies[identifiant]
            duree = 0.0
        else:
            debut = time.perf_counter()
            reponse = rag_pdf.demander(cas["question"])
            duree = time.perf_counter() - debut

        resultat = evaluer_reponse_textuelle(cas, reponse)
        resultat["duration_s"] = round(duree, 2)
        resultats.append(resultat)

    resume = calculer_resume(resultats)
    rapport = {
        "summary": resume,
        "results": resultats,
    }

    if args.json:
        print(json.dumps(rapport, ensure_ascii=False, indent=2))
    else:
        afficher_resultats(resultats, resume)

    if args.output:
        chemin_sortie = args.output
        if not chemin_sortie.is_absolute():
            chemin_sortie = BASE_DIR / chemin_sortie
        chemin_sortie.parent.mkdir(parents=True, exist_ok=True)
        chemin_sortie.write_text(
            json.dumps(rapport, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nRapport JSON sauvegarde : {chemin_sortie}")

    if (
        args.fail_under_success_rate is not None
        and resume["success_rate"] < args.fail_under_success_rate
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
