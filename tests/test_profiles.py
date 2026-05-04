import os
import unittest
from unittest.mock import patch

from rag.profiles import (
    charger_profile,
    lister_profils_disponibles,
    profil_domaine_actif,
    profil_est_actif,
)


class DomainProfileTests(unittest.TestCase):
    """Teste le chargement des profils sans toucher au moteur RAG."""

    def test_profil_generic_est_neutre(self):
        profil = charger_profile("generic")

        self.assertEqual(profil.name, "generic")
        self.assertTrue(profil.is_generic)
        self.assertEqual(
            profil.expand_question("What is the platform?"),
            "What is the platform?",
        )

    def test_profil_inconnu_retourne_generic(self):
        profil = charger_profile("profil-qui-n-existe-pas")

        self.assertEqual(profil.name, "generic")

    def test_profil_everpure_charge_les_regles_json(self):
        profil = charger_profile("everpure")

        self.assertEqual(profil.name, "everpure")
        self.assertFalse(profil.is_generic)
        self.assertGreater(len(profil.expansion_rules), 0)

        question_enrichie = profil.expand_question("What is useful for a GSI?")
        self.assertIn("global system integrator", question_enrichie)

    def test_liste_profils_contient_generic_et_everpure(self):
        profils = lister_profils_disponibles()

        self.assertIn("generic", profils)
        self.assertIn("everpure", profils)

    def test_profil_actif_vient_de_la_variable_environnement(self):
        with patch.dict(os.environ, {"RAG_DOMAIN_PROFILE": "everpure"}):
            self.assertEqual(profil_domaine_actif().name, "everpure")
            self.assertTrue(profil_est_actif("everpure"))

        with patch.dict(os.environ, {"RAG_DOMAIN_PROFILE": "generic"}):
            self.assertEqual(profil_domaine_actif().name, "generic")
            self.assertFalse(profil_est_actif("everpure"))


if __name__ == "__main__":
    unittest.main()
