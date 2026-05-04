import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_retrieval import (
    calculer_resume,
    charger_questions,
    sauvegarder_rapport,
    verifier_seuils,
)


class EvaluateRetrievalScriptTests(unittest.TestCase):
    """Teste les helpers du benchmark sans lancer FAISS ni OpenAI."""

    def test_charger_questions_valide_un_fichier_simple(self):
        donnees = [
            {
                "id": "sample",
                "question": "What is indexed?",
                "expected_docs": ["doc.pdf"],
                "expected_keywords": ["index"],
            }
        ]

        with tempfile.TemporaryDirectory() as dossier:
            chemin = Path(dossier) / "questions.json"
            chemin.write_text(json.dumps(donnees), encoding="utf-8")

            questions = charger_questions(chemin)

        self.assertEqual(questions[0]["id"], "sample")
        self.assertEqual(questions[0]["question"], "What is indexed?")

    def test_charger_questions_refuse_un_cas_sans_question(self):
        with tempfile.TemporaryDirectory() as dossier:
            chemin = Path(dossier) / "questions.json"
            chemin.write_text(json.dumps([{"id": "broken"}]), encoding="utf-8")

            with self.assertRaises(ValueError):
                charger_questions(chemin)

    def test_calculer_resume_agrege_les_scores(self):
        resume = calculer_resume(
            [
                {"doc_recall": 1.0, "keyword_recall": 0.5, "duration_s": 2.0},
                {"doc_recall": 0.5, "keyword_recall": 1.0, "duration_s": 4.0},
            ]
        )

        self.assertEqual(resume["questions"], 2)
        self.assertEqual(resume["mean_doc_recall"], 0.75)
        self.assertEqual(resume["mean_keyword_recall"], 0.75)
        self.assertEqual(resume["mean_duration_s"], 3.0)

    def test_verifier_seuils_accepte_un_resume_au_dessus_des_seuils(self):
        resume = {"mean_doc_recall": 0.8, "mean_keyword_recall": 0.9}

        with contextlib.redirect_stderr(io.StringIO()):
            resultat = verifier_seuils(
                resume,
                resultats=[],
                fail_under_doc_recall=0.7,
                fail_under_keyword_recall=0.8,
            )

        self.assertTrue(resultat)

    def test_verifier_seuils_refuse_un_resume_sous_un_seuil(self):
        resume = {"mean_doc_recall": 0.6, "mean_keyword_recall": 0.9}

        with contextlib.redirect_stderr(io.StringIO()):
            resultat = verifier_seuils(
                resume,
                resultats=[],
                fail_under_doc_recall=0.7,
                fail_under_keyword_recall=0.8,
            )

        self.assertFalse(resultat)

    def test_verifier_seuils_peut_echouer_sur_une_question_ratee(self):
        resume = {"mean_doc_recall": 0.8, "mean_keyword_recall": 0.8}
        resultats = [
            {"id": "ok", "doc_recall": 1.0, "keyword_recall": 1.0},
            {"id": "miss", "doc_recall": 0.0, "keyword_recall": 1.0},
        ]

        with contextlib.redirect_stderr(io.StringIO()):
            resultat = verifier_seuils(
                resume,
                resultats=resultats,
                fail_under_doc_recall=None,
                fail_under_keyword_recall=None,
                fail_on_any_miss=True,
            )

        self.assertFalse(resultat)

    def test_sauvegarder_rapport_ecrit_configuration_resume_et_resultats(self):
        configuration = {"top_k": 5}
        resume = {"mean_doc_recall": 1.0}
        resultats = [{"id": "sample"}]

        with tempfile.TemporaryDirectory() as dossier:
            chemin = Path(dossier) / "rapport.json"
            sauvegarder_rapport(chemin, configuration, resultats, resume)
            rapport = json.loads(chemin.read_text(encoding="utf-8"))

        self.assertEqual(rapport["configuration"], configuration)
        self.assertEqual(rapport["summary"], resume)
        self.assertEqual(rapport["results"], resultats)


if __name__ == "__main__":
    unittest.main()
