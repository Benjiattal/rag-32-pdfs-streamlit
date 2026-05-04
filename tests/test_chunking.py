import unittest

from rag.chunking import (
    decouper_en_phrases,
    decouper_phrase_trop_longue,
    decouper_texte,
    decouper_texte_par_sections,
)


class ChunkingTests(unittest.TestCase):
    """Teste le decoupage intelligent sans dependance externe."""

    def test_decoupe_en_phrases(self):
        texte = "First sentence. Second sentence! Third sentence?"

        phrases = decouper_en_phrases(texte)

        self.assertEqual(
            phrases,
            ["First sentence.", "Second sentence!", "Third sentence?"],
        )

    def test_decoupe_avec_chevauchement_de_phrase(self):
        texte = "Alpha one. Beta two. Gamma three. Delta four."

        chunks = decouper_texte(
            texte,
            taille=24,
            chevauchement_phrases=1,
        )

        self.assertGreaterEqual(len(chunks), 3)
        self.assertIn("Alpha one.", chunks[0])
        self.assertIn("Beta two.", chunks[0])
        self.assertIn("Beta two.", chunks[1])
        self.assertIn("Gamma three.", chunks[1])

    def test_decoupe_phrase_trop_longue_en_secours(self):
        phrase = "abcdefghijklmnopqrstuvwxyz"

        chunks = decouper_phrase_trop_longue(phrase, taille=10)

        self.assertEqual(chunks, ["abcdefghij", "klmnopqrst", "uvwxyz"])

    def test_decoupe_par_sections_garde_le_titre(self):
        texte = """
Overview
This platform indexes documents and answers with sources.

Technical Details
Embeddings are stored in FAISS. BM25 improves exact matches.
"""

        chunks = decouper_texte_par_sections(
            texte,
            taille=120,
            chevauchement_phrases=1,
        )

        titres = [titre for _, titre in chunks]
        textes = [chunk for chunk, _ in chunks]

        self.assertIn("Overview", titres)
        self.assertIn("Technical Details", titres)
        self.assertTrue(any("TITRE DE SECTION: Overview" in chunk for chunk in textes))


if __name__ == "__main__":
    unittest.main()
