import unittest

from rag.retrieval import (
    normaliser_texte_recherche,
    normaliser_mots,
    score_lexical,
    scores_bm25,
    tokeniser_recherche,
)


class RetrievalLexicalTests(unittest.TestCase):
    """Teste les briques lexicales utilisees par BM25 et le reranking hybride."""

    def test_normalisation_retire_accents_et_minuscules(self):
        texte = normaliser_texte_recherche("Électrique FlashArray")

        self.assertEqual(texte, "electrique flasharray")

    def test_tokenisation_garde_les_noms_produits(self):
        tokens = tokeniser_recherche("FlashArray//XL power: 2,635 watts")

        self.assertIn("flasharray//xl", tokens)
        self.assertIn("power", tokens)
        self.assertIn("watts", tokens)

    def test_normaliser_mots_retire_mots_vides(self):
        mots = normaliser_mots("the FlashArray with power specs")

        self.assertIn("flasharray", mots)
        self.assertIn("power", mots)
        self.assertNotIn("the", mots)
        self.assertNotIn("with", mots)

    def test_score_lexical_mesure_les_mots_communs(self):
        score = score_lexical(
            "FlashArray power consumption",
            "The FlashArray datasheet lists power and cooling values.",
        )

        self.assertGreater(score, 0.5)

    def test_bm25_favorise_le_document_le_plus_pertinent(self):
        textes = [
            "FlashArray power consumption watts cooling technical specifications.",
            "General partner program and business benefits.",
            "FlashBlade object storage overview.",
        ]

        scores = scores_bm25("FlashArray power watts", textes)

        self.assertEqual(len(scores), 3)
        self.assertEqual(max(scores), scores[0])
        self.assertAlmostEqual(scores[0], 1.0)
        self.assertGreater(scores[0], scores[1])
        self.assertGreater(scores[0], scores[2])


if __name__ == "__main__":
    unittest.main()
