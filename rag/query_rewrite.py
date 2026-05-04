"""
Reformulation de requete pour le retrieval.

Responsabilite :
produire une reformulation courte de la question utilisateur pour ameliorer le
retrieval, sans jamais rediger la reponse finale.

Pourquoi isoler ce module ?
- La reformulation est optionnelle et appelle un LLM distant.
- Son cache depend du modele et du profil de domaine actif.
- Le retrieval local peut rester testable sans appel reseau.
"""

from __future__ import annotations

import hashlib
import os
import pickle

from rag.config import (
    INDEX_DIR,
    QUERY_REWRITE_CACHE_FILE,
    DEFAULT_CHAT_MODEL,
    QUERY_REWRITE_PROMPT_VERSION,
)
from rag.llm import creer_client_openai
from rag.profiles import profil_domaine_actif
from rag.retrieval import (
    construire_question_recherche as construire_question_recherche_locale,
    construire_requetes_recherche as construire_requetes_recherche_locales,
)

def cle_cache_query_rewrite(question: str, modele: str) -> str:
    """
    Calcule la clé du cache de query rewriting.

    Comme pour les embeddings, on inclut le modèle dans la clé : deux modèles
    peuvent produire des reformulations différentes.
    """
    profil = profil_domaine_actif()
    contenu = f"{QUERY_REWRITE_PROMPT_VERSION}\0{profil.name}\0{modele}\0{question}".encode("utf-8")
    return hashlib.sha256(contenu).hexdigest()


def charger_cache_query_rewrite() -> dict[str, str]:
    """
    Charge le cache des reformulations LLM.

    Ce cache évite de rappeler le modèle pour la même question, notamment quand
    Streamlit réexécute le script ou quand le mode debug réaffiche la requête.
    """
    if not QUERY_REWRITE_CACHE_FILE.exists():
        return {}

    try:
        with QUERY_REWRITE_CACHE_FILE.open("rb") as fichier:
            cache = pickle.load(fichier)
    except (OSError, EOFError, pickle.UnpicklingError):
        return {}

    if not isinstance(cache, dict):
        return {}

    return cache


def sauvegarder_cache_query_rewrite(cache: dict[str, str]) -> None:
    """Sauvegarde atomiquement le cache des reformulations LLM."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    fichier_temporaire = QUERY_REWRITE_CACHE_FILE.with_suffix(".pkl.tmp")

    with fichier_temporaire.open("wb") as fichier:
        pickle.dump(cache, fichier)

    os.replace(fichier_temporaire, QUERY_REWRITE_CACHE_FILE)


def reecrire_question_avec_llm(question: str) -> str:
    """
    Demande au LLM une reformulation courte pour améliorer le retrieval.

    Ce n'est PAS la réponse finale.
    Le LLM sert ici de traducteur d'intention :
    - corriger quelques fautes ;
    - ajouter des synonymes anglais/français utiles aux PDF ;
    - expliciter les acronymes ;
    - conserver strictement le sens de la question.

    Sécurité :
    si l'appel OpenAI échoue, on retourne une chaîne vide et le RAG continue avec
    l'expansion déterministe existante.
    """
    modele = os.getenv("OPENAI_QUERY_REWRITE_MODEL", os.getenv("OPENAI_MODEL", DEFAULT_CHAT_MODEL))
    cle_cache = cle_cache_query_rewrite(question, modele)
    cache = charger_cache_query_rewrite()

    if cle_cache in cache:
        return cache[cle_cache]

    profil = profil_domaine_actif()
    contraintes_profil = profil.query_rewrite_extra_text()
    if contraintes_profil:
        contraintes_profil = "\n" + contraintes_profil

    prompt = f"""
{profil.query_rewrite_context}

Tache :
Reformule la question utilisateur pour la recherche documentaire uniquement.

Contraintes :
- Ne réponds pas à la question.
- Ne change pas l'intention.
- Corrige les fautes évidentes.
- Ajoute des synonymes utiles en français et en anglais.
- Ajoute les noms produits probables seulement s'ils sont déjà implicites dans la question.
- N'ajoute pas de concepts trop larges comme alternatives, concurrents, marché, stratégie ou généralités.
- Réponse en une seule ligne.
- Maximum 45 mots.{contraintes_profil}

Question utilisateur :
{question}
""".strip()

    try:
        client = creer_client_openai()
        reponse = client.responses.create(
            model=modele,
            input=prompt,
            temperature=0,
        )
    except Exception:
        return ""

    rewrite = " ".join(reponse.output_text.split())

    # On limite volontairement la taille : une reformulation trop longue dilue la
    # recherche et peut faire remonter du bruit.
    if len(rewrite) > 600:
        rewrite = rewrite[:600]

    cache[cle_cache] = rewrite
    sauvegarder_cache_query_rewrite(cache)

    return rewrite


def construire_question_recherche(
    question: str,
    utiliser_query_rewrite_llm: bool | None = None,
) -> str:
    """
    Wrapper historique qui injecte la reformulation LLM dans le retrieval local.

    La mecanique de construction vit maintenant dans `rag.retrieval`. L'appel
    OpenAI reste ici pour eviter les imports circulaires et garder une separation
    claire entre logique locale et appel reseau.
    """
    return construire_question_recherche_locale(
        question,
        utiliser_query_rewrite_llm=utiliser_query_rewrite_llm,
        rewrite_fn=reecrire_question_avec_llm,
    )


def construire_requetes_recherche(
    question: str,
    utiliser_query_rewrite_llm: bool | None = None,
) -> list[tuple[str, str]]:
    """
    Wrapper historique pour construire les requetes avec query rewrite LLM optionnel.
    """
    return construire_requetes_recherche_locales(
        question,
        utiliser_query_rewrite_llm=utiliser_query_rewrite_llm,
        rewrite_fn=reecrire_question_avec_llm,
    )



def statistiques_cache_query_rewrite() -> dict[str, int | bool]:
    """Retourne quelques infos sur le cache de query rewriting."""
    cache = charger_cache_query_rewrite()

    return {
        "cache_existe": QUERY_REWRITE_CACHE_FILE.exists(),
        "nombre_rewrites": len(cache),
        "taille_octets": QUERY_REWRITE_CACHE_FILE.stat().st_size
        if QUERY_REWRITE_CACHE_FILE.exists()
        else 0,
    }


