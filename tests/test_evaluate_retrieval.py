import json
import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_retrieval import calculer_resume, charger_questions


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


if __name__ == "__main__":
    unittest.main()
