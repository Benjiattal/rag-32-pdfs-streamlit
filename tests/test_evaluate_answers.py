import unittest

from scripts.evaluate_answers import (
    calculer_resume,
    contient_citation,
    evaluer_reponse_textuelle,
    reponse_dit_ne_pas_savoir,
)


class EvaluateAnswersTests(unittest.TestCase):
    """Teste l'evaluation de réponse finale sans appeler OpenAI."""

    def test_detecte_citation(self):
        self.assertTrue(contient_citation("The answer is sourced [1]."))
        self.assertFalse(contient_citation("The answer has no source."))

    def test_detecte_reponse_inconnue(self):
        self.assertTrue(reponse_dit_ne_pas_savoir("Je ne sais pas."))
        self.assertTrue(reponse_dit_ne_pas_savoir("I don't know from the context."))

    def test_evalue_faits_citations_et_interdits(self):
        cas = {
            "id": "sample",
            "question": "What does Acme do?",
            "expected_facts": ["search", "citations"],
            "forbidden_terms": ["blockchain"],
            "must_have_citation": True,
        }
        reponse = "Acme provides document search with citations [1]."

        resultat = evaluer_reponse_textuelle(cas, reponse)

        self.assertTrue(resultat["ok"])
        self.assertEqual(resultat["facts_recall"], 1.0)

    def test_echec_si_hallucination_interdite(self):
        cas = {
            "question": "What does Acme do?",
            "expected_facts": ["search"],
            "forbidden_terms": ["blockchain"],
        }
        resultat = evaluer_reponse_textuelle(
            cas,
            "Acme provides search and blockchain automation.",
        )

        self.assertFalse(resultat["ok"])
        self.assertIn("blockchain", resultat["forbidden_terms_present"])

    def test_resume(self):
        resume = calculer_resume(
            [
                {"ok": True, "facts_recall": 1.0},
                {"ok": False, "facts_recall": 0.5},
            ]
        )

        self.assertEqual(resume["questions"], 2)
        self.assertEqual(resume["success_rate"], 0.5)
        self.assertEqual(resume["mean_facts_recall"], 0.75)


if __name__ == "__main__":
    unittest.main()
