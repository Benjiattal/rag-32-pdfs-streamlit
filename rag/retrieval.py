"""
Recherche documentaire.

Responsabilite :
retrouver les meilleurs chunks a partir d'une question, appliquer les filtres
metadata, adapter top_k/candidate_k et gerer les cas techniques.
"""

from rag.engine import (  # noqa: F401
    normaliser_texte_recherche,
    tokeniser_recherche,
    normaliser_mots,
    score_lexical,
    scores_bm25,
    enrichir_question_pour_recherche,
    ajouter_requete_unique,
    construire_requetes_recherche,
    construire_question_recherche,
    fusionner_candidats_multi_requetes,
    question_demande_valeurs_techniques,
    question_demande_portefeuille_capacites,
    question_demande_inventaire_flasharray,
    question_demande_inventaire_flashblade,
    calculer_parametres_recherche_adaptatifs,
    diversifier_resultats_par_document,
    contient_valeurs_techniques,
    filtrer_resultats_pour_valeurs_techniques,
    rechercher,
)
