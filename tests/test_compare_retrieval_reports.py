import unittest

from scripts.compare_retrieval_reports import comparer_resumes, formater_delta


class CompareRetrievalReportsTests(unittest.TestCase):
    """Teste la comparaison de rapports sans relancer le benchmark."""

    def test_formater_delta_affiche_un_signe(self):
        self.assertEqual(formater_delta(0.06), "+0.06")
        self.assertEqual(formater_delta(-0.05), "-0.05")
        self.assertEqual(formater_delta(0.6, "s"), "+0.60s")

    def test_comparer_resumes_calcule_les_ecarts(self):
        lignes = comparer_resumes(
            {
                "mean_doc_recall": 0.82,
                "mean_keyword_recall": 0.76,
                "mean_duration_s": 1.2,
            },
            {
                "mean_doc_recall": 0.88,
                "mean_keyword_recall": 0.71,
                "mean_duration_s": 1.8,
            },
        )

        deltas = {ligne["metric"]: round(ligne["delta"], 2) for ligne in lignes}

        self.assertEqual(deltas["doc_recall"], 0.06)
        self.assertEqual(deltas["keyword_recall"], -0.05)
        self.assertEqual(deltas["latency"], 0.6)


if __name__ == "__main__":
    unittest.main()
