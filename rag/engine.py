"""
==============================
SCRIPT RAG PDF - VERSION COMMENTEE
==============================

Ce fichier contient un petit RAG local pour interroger quelques PDF.

RAG signifie "Retrieval Augmented Generation".
En francais simple :

1. On lit les PDF.
2. On decoupe leur texte en petits morceaux.
3. On transforme chaque morceau en vecteur avec OpenAI.
4. On stocke ces vecteurs dans une base vectorielle FAISS.
5. Quand on pose une question, FAISS retrouve les morceaux les plus proches.
6. On donne ces morceaux au modele OpenAI pour produire une reponse.

Pourquoi ce fichier est tres commente ?
Parce que l'objectif est d'apprendre. Les commentaires expliquent autant le
"pourquoi" que le "comment".
"""

from __future__ import annotations

# ==============================
# --- 1. Imports ---
# ==============================

import argparse  # interface en ligne de commande
import hashlib  # cree une empreinte stable pour le cache d'embeddings
import importlib.util  # verification legere des dependances optionnelles
import json  # lecture stricte des reponses JSON du reranker LLM
import os  # lecture des variables d'environnement
import pickle  # sauvegarde / chargement d'objets Python
import re  # decoupage de texte et nettoyage simple
import select  # timeout de lecture du worker BGE
import subprocess  # lance BGE dans un worker separe pour stabiliser Streamlit
import time  # petites pauses lors des retries de lecture disque
from pathlib import Path  # gestion propre des chemins fichiers

from rag.config import (
    BASE_DIR,
    PDF_DIR,
    INDEX_DIR,
    WEB_DIR,
    FAISS_INDEX_FILE,
    METADATA_FILE,
    WEB_CHUNKS_FILE,
    EMBEDDING_CACHE_FILE,
    QUERY_REWRITE_CACHE_FILE,
    LLM_RERANK_CACHE_FILE,
    DEFAULT_CHAT_MODEL,
    DEFAULT_CHAT_MODELS,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_TOP_K,
    DEFAULT_CANDIDATE_K,
    DEFAULT_CONTEXT_TOKEN_BUDGET,
    DEFAULT_CHUNK_SIZE_CHARS,
    DEFAULT_CHUNK_MIN_CHARS,
    DEFAULT_CHUNK_OVERLAP_SENTENCES,
    DEFAULT_MIN_SCORE,
    DEFAULT_TEMPERATURE,
    DEFAULT_ALLOWED_WEB_DOMAINS,
    DEFAULT_QUERY_REWRITE_LLM,
    DEFAULT_LLM_RERANKER,
    DEFAULT_LLM_RERANK_CANDIDATES,
    DEFAULT_CROSS_ENCODER_RERANKER,
    DEFAULT_CROSS_ENCODER_MODEL,
    DEFAULT_CROSS_ENCODER_RERANK_CANDIDATES,
    DEFAULT_BGE_WORKER_TIMEOUT_SECONDS,
    QUERY_REWRITE_PROMPT_VERSION,
    LLM_RERANK_PROMPT_VERSION,
    POIDS_SCORE_SEMANTIQUE,
    POIDS_SCORE_BM25,
)
from rag.models import (
    Morceau,
    ResultatRecherche,
    FiltreMetadata,
    ParametresRecherche,
)
from rag.llm import appeler_modele, creer_client_openai
from rag.env import charger_env_local
from rag.embeddings import (
    cle_cache_embedding,
    charger_cache_embeddings,
    sauvegarder_cache_embeddings,
    statistiques_cache_embeddings,
    creer_embeddings,
)
from rag.ingestion import (
    lister_pdfs,
    domaines_web_autorises,
    domaine_est_autorise,
    lister_sources_web,
    lister_documents_indexes,
    statistiques_index,
    UnpicklerCompatibleRag,
    charger_morceaux_pickle,
    charger_web_chunks,
    sauvegarder_web_chunks,
    lire_pdf,
    extraire_tableaux_page,
    convertir_tableau_en_markdown,
    extraire_extraits_techniques,
    lire_page_web,
    indexer_pages_web,
    indexer_pdf,
    charger_index,
)
from rag.retrieval import (
    normaliser_texte_recherche,
    tokeniser_recherche,
    normaliser_mots,
    score_lexical,
    scores_bm25,
    enrichir_question_pour_recherche,
    ajouter_requete_unique,
    construire_question_recherche as construire_question_recherche_locale,
    construire_requetes_recherche as construire_requetes_recherche_locales,
    fusionner_candidats_multi_requetes,
    diversifier_resultats_par_document,
    contient_valeurs_techniques,
    filtrer_resultats_pour_valeurs_techniques as filtrer_resultats_pour_valeurs_techniques_local,
)
from rag.profiles import (
    lister_profils_disponibles,
    profil_domaine_actif,
    profil_est_actif,
)
from rag.everpure import (
    prioriser_inventaires_produits as prioriser_inventaires_produits_everpure,
    question_demande_inventaire_produit as question_demande_inventaire_produit_everpure,
)
from rag.prompting import (
    construire_contexte_long,
    construire_prompt,
    estimer_tokens,
)



# Pourquoi separer configuration, modeles et moteur ?
# -> `config.py` porte les chemins et valeurs par defaut.
# -> `models.py` porte les dataclasses partagees.
# -> `engine.py` garde l'orchestration RAG historique.


def variable_env_booleenne(nom: str, valeur_defaut: bool) -> bool:
    """
    Lit une variable d'environnement comme un booléen.

    Exemples acceptés :
    - vrai : "1", "true", "yes", "oui", "on"
    - faux : "0", "false", "no", "non", "off"

    C'est pratique pour activer/désactiver des options sans modifier le code.
    """
    valeur = os.getenv(nom)

    if valeur is None:
        return valeur_defaut

    return valeur.strip().lower() in {"1", "true", "yes", "oui", "on"}


# Le chunking vit dans un module dedie : c'est une brique autonome du RAG.
# On garde ces imports dans engine.py pour conserver l'API historique.
from rag.chunking import (
    decouper_texte,
    decouper_texte_par_sections,
    extraire_sections_simples,
    ligne_parasite_pdf,
    est_titre_section_probable,
    nettoyer_titre_section,
    decouper_en_phrases,
    decouper_phrase_trop_longue,
)


# ==============================
# --- 7. Ingestion et index FAISS ---
# ==============================

# L'ingestion PDF/Web et le chargement FAISS vivent dans `rag.ingestion`.
# Les fonctions sont importees ci-dessus pour conserver l'API historique de
# `rag_pdf.py` et eviter une regression cote Streamlit/CLI.


# ==============================
# --- 8. Recherche ---
# ==============================


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


def cle_cache_llm_rerank(
    question: str,
    resultats: list[ResultatRecherche],
    modele: str,
    top_k: int,
) -> str:
    """
    Calcule une clé stable pour le cache du reranker LLM.

    La clé inclut :
    - la question ;
    - le modèle de reranking ;
    - les candidats vus par le LLM ;
    - le nombre de résultats demandés.

    Ainsi, si l'index ou les chunks changent, le cache ne réutilise pas un ancien
    classement devenu incohérent.
    """
    morceaux = []

    for resultat in resultats:
        morceau = resultat.morceau
        empreinte_texte = hashlib.sha256(morceau.texte.encode("utf-8")).hexdigest()[:16]
        morceaux.append(
            f"{morceau.fichier}|{morceau.page}|{getattr(morceau, 'numero', '?')}|{empreinte_texte}"
        )

    contenu = (
        f"{LLM_RERANK_PROMPT_VERSION}\0{modele}\0{top_k}\0{question}\0"
        + "\0".join(morceaux)
    ).encode("utf-8")
    return hashlib.sha256(contenu).hexdigest()


def charger_cache_llm_rerank() -> dict[str, list[int]]:
    """
    Charge le cache des classements LLM.

    Le cache stocke uniquement l'ordre choisi par le LLM, pas de données
    sensibles. Si le fichier est absent ou illisible, on repart simplement de
    zéro.
    """
    if not LLM_RERANK_CACHE_FILE.exists():
        return {}

    try:
        with LLM_RERANK_CACHE_FILE.open("rb") as fichier:
            cache = pickle.load(fichier)
    except (OSError, EOFError, pickle.UnpicklingError):
        return {}

    if not isinstance(cache, dict):
        return {}

    return cache


def sauvegarder_cache_llm_rerank(cache: dict[str, list[int]]) -> None:
    """Sauvegarde atomiquement le cache du reranker LLM."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    fichier_temporaire = LLM_RERANK_CACHE_FILE.with_suffix(".pkl.tmp")

    with fichier_temporaire.open("wb") as fichier:
        pickle.dump(cache, fichier)

    os.replace(fichier_temporaire, LLM_RERANK_CACHE_FILE)


def statistiques_cache_llm_rerank() -> dict[str, int | bool]:
    """Retourne quelques infos sur le cache du reranker LLM."""
    cache = charger_cache_llm_rerank()

    return {
        "cache_existe": LLM_RERANK_CACHE_FILE.exists(),
        "nombre_reranks": len(cache),
        "taille_octets": LLM_RERANK_CACHE_FILE.stat().st_size
        if LLM_RERANK_CACHE_FILE.exists()
        else 0,
    }


def extraire_liste_json_entiers(texte: str) -> list[int]:
    """
    Extrait une liste d'entiers depuis une réponse LLM.

    On demande au modèle de répondre en JSON, mais cette fonction reste tolérante
    si le modèle ajoute quelques mots autour.
    """
    match = re.search(r"\[[^\]]*\]", texte, flags=re.DOTALL)

    if not match:
        return []

    try:
        valeurs = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []

    return [int(valeur) for valeur in valeurs if isinstance(valeur, int)]


def reranker_resultats_avec_llm(
    question: str,
    resultats: list[ResultatRecherche],
    top_k: int,
) -> list[ResultatRecherche]:
    """
    Reclasse les meilleurs candidats avec un LLM.

    Différence avec FAISS/BM25 :
    - FAISS compare des vecteurs ;
    - BM25 compare des mots ;
    - le LLM juge si un passage aide vraiment à répondre à la question.

    On ne lui demande pas de rédiger la réponse finale. Il retourne seulement une
    liste de numéros de candidats, par ordre d'utilité.
    """
    if len(resultats) <= top_k:
        return resultats

    modele = os.getenv("OPENAI_RERANK_MODEL", os.getenv("OPENAI_MODEL", DEFAULT_CHAT_MODEL))
    limite_candidats = int(
        os.getenv("RAG_LLM_RERANK_CANDIDATES", str(DEFAULT_LLM_RERANK_CANDIDATES))
    )
    candidats = resultats[: max(top_k, limite_candidats)]
    cle_cache = cle_cache_llm_rerank(question, candidats, modele, top_k)
    cache = charger_cache_llm_rerank()

    if cle_cache in cache:
        ordre = cache[cle_cache]
    else:
        blocs = []

        for numero, resultat in enumerate(candidats, start=1):
            morceau = resultat.morceau
            extrait = " ".join(morceau.texte.split())[:900]
            blocs.append(
                f"{numero}. Document: {morceau.fichier} | page {morceau.page} | "
                f"score hybride {resultat.score_final:.3f}\n{extrait}"
            )

        prompt = f"""
Tu es un reranker RAG.

Question utilisateur :
{question}

Tache :
Classe les passages ci-dessous selon leur utilite pour repondre a la question.

Regles :
- Ne redige pas la reponse finale.
- Ne selectionne que les passages vraiment utiles.
- Favorise les passages factuels, tableaux, specs et definitions directes.
- Si la question demande une liste de produits, favorise les passages qui couvrent plusieurs produits ou familles.
- Retourne uniquement un tableau JSON d'entiers, par exemple [3, 1, 7].
- Maximum {top_k} numeros.

Passages candidats :

{chr(10).join(blocs)}
""".strip()

        try:
            client = creer_client_openai()
            reponse = client.responses.create(
                model=modele,
                input=prompt,
                temperature=0,
            )
            ordre = extraire_liste_json_entiers(reponse.output_text)
        except Exception:
            return resultats

        cache[cle_cache] = ordre
        sauvegarder_cache_llm_rerank(cache)

    selectionnes: list[ResultatRecherche] = []
    numeros_deja_vus: set[int] = set()

    for numero in ordre:
        if numero < 1 or numero > len(candidats) or numero in numeros_deja_vus:
            continue

        resultat = candidats[numero - 1]
        # Petit bonus pour refléter le jugement LLM sans perdre les scores de
        # debug FAISS/BM25. Le premier choix reçoit le bonus le plus fort.
        bonus = 0.30 / max(1, len(selectionnes) + 1)
        selectionnes.append(
            ResultatRecherche(
                morceau=resultat.morceau,
                score_semantique=resultat.score_semantique,
                score_lexical=resultat.score_lexical,
                score_final=resultat.score_final + bonus,
            )
        )
        numeros_deja_vus.add(numero)

    if not selectionnes:
        return resultats

    restants = [resultat for index, resultat in enumerate(candidats, start=1) if index not in numeros_deja_vus]
    hors_fenetre = resultats[len(candidats) :]

    return selectionnes + restants + hors_fenetre


_CROSS_ENCODER_MODEL_CACHE = None
_CROSS_ENCODER_MODEL_NAME_CACHE = ""
_BGE_WORKER_PROCESS = None
_BGE_WORKER_MODEL = ""
_DERNIER_BGE_ACTIF = False
# Timeout court : si BGE n'est pas prêt, on préfère répondre vite avec
# FAISS + BM25 plutôt que bloquer l'interface pour un gain inexistant.
DEFAULT_BGE_WORKER_TIMEOUT_SECONDS = 5


def charger_cross_encoder():
    """
    Charge le modèle cross-encoder local une seule fois.

    Difference avec FAISS :
    - FAISS encode la question et le chunk separement, puis compare deux vecteurs ;
    - un cross-encoder lit la paire (question, chunk) ensemble et donne un score
      de pertinence direct.

    C'est souvent meilleur pour départager les 20 meilleurs candidats, mais plus
    lourd :
    - dependance `sentence-transformers` ;
    - telechargement du modele au premier usage ;
    - inference locale plus lente que FAISS/BM25.

    Pour cette raison, il reste optionnel.
    """
    global _CROSS_ENCODER_MODEL_CACHE, _CROSS_ENCODER_MODEL_NAME_CACHE

    modele = os.getenv("RAG_CROSS_ENCODER_MODEL", DEFAULT_CROSS_ENCODER_MODEL)

    if _CROSS_ENCODER_MODEL_CACHE is not None and _CROSS_ENCODER_MODEL_NAME_CACHE == modele:
        return _CROSS_ENCODER_MODEL_CACHE

    try:
        from sentence_transformers import CrossEncoder
    except ModuleNotFoundError as erreur:
        raise ModuleNotFoundError(
            "Le reranker cross-encoder nécessite sentence-transformers. "
            "Installe-le avec : pip install sentence-transformers"
        ) from erreur

    _CROSS_ENCODER_MODEL_CACHE = CrossEncoder(modele)
    _CROSS_ENCODER_MODEL_NAME_CACHE = modele
    return _CROSS_ENCODER_MODEL_CACHE


def chemin_worker_bge_python() -> str:
    """
    Retourne le Python utilisé pour le worker BGE externe.

    Principe :
    - Streamlit reste dans un environnement léger et stable ;
    - BGE tourne dans `.venv_bge`, avec `sentence-transformers` et PyTorch ;
    - si tu veux un autre environnement, définis `RAG_BGE_WORKER_PYTHON`.
    """
    chemin_env = os.getenv("RAG_BGE_WORKER_PYTHON")
    if chemin_env:
        return chemin_env

    chemin_venv311 = BASE_DIR / ".venv311" / "bin" / "python"
    if chemin_venv311.exists():
        return str(chemin_venv311)

    chemin_bge_historique = BASE_DIR / ".venv_bge" / "bin" / "python"
    return str(chemin_bge_historique)


def bge_worker_externe_disponible() -> bool:
    """Vérifie si le worker BGE séparé peut être lancé."""
    return Path(chemin_worker_bge_python()).exists() and (
        BASE_DIR / "scripts" / "bge_worker.py"
    ).exists()


def dernier_reranker_bge_actif() -> bool:
    """
    Indique si BGE a réellement reranké la dernière recherche.

    C'est différent de "BGE est coché dans l'UI" :
    - coché = l'utilisateur demande BGE ;
    - actif = le worker/model a répondu et ses scores ont été utilisés.
    """
    return _DERNIER_BGE_ACTIF


def cross_encoder_disponible() -> bool:
    """
    Indique si la dependance Python du cross-encoder est installee.

    On n'importe pas `sentence_transformers` ici : son import charge beaucoup de
    dependances ML. Pour l'interface Streamlit, une simple verification de
    presence suffit et evite de ralentir le demarrage.

    Garde-fou :
    sur Python 3.13/macOS, le stack `sentence-transformers` peut bloquer au
    chargement dans cet environnement. On exige donc une variable explicite avant
    d'autoriser l'UI a l'utiliser :

    RAG_ENABLE_BGE_EXPERIMENTAL=1
    """
    if bge_worker_externe_disponible():
        return True

    if not variable_env_booleenne("RAG_ENABLE_BGE_EXPERIMENTAL", False):
        return False

    return importlib.util.find_spec("sentence_transformers") is not None


def charger_worker_bge_externe():
    """
    Lance ou réutilise le worker BGE externe.

    Le premier appel peut prendre quelques dizaines de secondes car le modèle est
    chargé en mémoire. Les appels suivants réutilisent le même processus, donc le
    reranking devient nettement plus rapide et surtout beaucoup plus stable pour
    Streamlit.
    """
    global _BGE_WORKER_PROCESS, _BGE_WORKER_MODEL

    modele = os.getenv("RAG_CROSS_ENCODER_MODEL", DEFAULT_CROSS_ENCODER_MODEL)
    worker_script = BASE_DIR / "scripts" / "bge_worker.py"
    python_worker = chemin_worker_bge_python()

    if (
        _BGE_WORKER_PROCESS is not None
        and _BGE_WORKER_PROCESS.poll() is None
        and _BGE_WORKER_MODEL == modele
    ):
        return _BGE_WORKER_PROCESS

    if not bge_worker_externe_disponible():
        return None

    env = os.environ.copy()
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env["RAG_CROSS_ENCODER_MODEL"] = modele

    processus = subprocess.Popen(
        [python_worker, str(worker_script)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        env=env,
    )

    timeout = float(os.getenv("RAG_BGE_WORKER_TIMEOUT", str(DEFAULT_BGE_WORKER_TIMEOUT_SECONDS)))

    if not processus.stdout:
        processus.kill()
        return None

    pret, _, _ = select.select([processus.stdout], [], [], timeout)
    if not pret:
        processus.kill()
        return None

    ligne_ready = processus.stdout.readline()
    try:
        ready = json.loads(ligne_ready)
    except json.JSONDecodeError:
        processus.kill()
        return None

    if ready.get("status") != "ready":
        processus.kill()
        return None

    _BGE_WORKER_PROCESS = processus
    _BGE_WORKER_MODEL = modele
    return processus


def scores_bge_externe(question: str, candidats: list[ResultatRecherche]) -> list[float] | None:
    """Demande les scores au worker BGE séparé."""
    processus = charger_worker_bge_externe()

    if not processus or not processus.stdin or not processus.stdout:
        return None

    payload = {
        "question": question,
        "candidates": [
            {
                "text": resultat.morceau.texte,
            }
            for resultat in candidats
        ],
    }

    timeout = float(os.getenv("RAG_BGE_WORKER_TIMEOUT", str(DEFAULT_BGE_WORKER_TIMEOUT_SECONDS)))

    try:
        processus.stdin.write(json.dumps(payload) + "\n")
        processus.stdin.flush()
        pret, _, _ = select.select([processus.stdout], [], [], timeout)
        if not pret:
            processus.kill()
            return None
        reponse = json.loads(processus.stdout.readline())
    except Exception:
        return None

    if reponse.get("status") != "ok":
        return None

    return [float(score) for score in reponse.get("scores", [])]


def reranker_resultats_avec_cross_encoder(
    question: str,
    resultats: list[ResultatRecherche],
    top_k: int,
) -> list[ResultatRecherche]:
    """
    Reclasse les candidats avec un vrai reranker cross-encoder.

    Pipeline :
    1. FAISS + BM25 produit une liste large de candidats.
    2. Le cross-encoder score les paires (question, chunk).
    3. On trie selon ce score, puis on garde les autres candidats derriere.

    On limite volontairement le nombre de candidats, car un cross-encoder lit le
    texte complet de chaque paire et coute beaucoup plus cher qu'un produit
    scalaire FAISS.
    """
    global _DERNIER_BGE_ACTIF

    _DERNIER_BGE_ACTIF = False

    if len(resultats) <= top_k:
        return resultats

    if not cross_encoder_disponible():
        return resultats

    limite_candidats = int(
        os.getenv(
            "RAG_CROSS_ENCODER_RERANK_CANDIDATES",
            str(DEFAULT_CROSS_ENCODER_RERANK_CANDIDATES),
        )
    )
    candidats = resultats[: max(top_k, limite_candidats)]

    scores = scores_bge_externe(question, candidats)

    if scores is None:
        try:
            modele = charger_cross_encoder()
            paires = [
                (question, " ".join(resultat.morceau.texte.split())[:1800])
                for resultat in candidats
            ]
            scores = modele.predict(paires)
        except Exception:
            # Le RAG doit rester utilisable meme si le modele local n'est pas encore
            # installe, telecharge ou compatible avec la machine.
            return resultats

    _DERNIER_BGE_ACTIF = True

    resultats_scores = []
    for resultat, score_cross_encoder in zip(candidats, scores):
        score_float = float(score_cross_encoder)
        resultats_scores.append(
            ResultatRecherche(
                morceau=resultat.morceau,
                score_semantique=resultat.score_semantique,
                score_lexical=resultat.score_lexical,
                # Le score cross-encoder peut etre negatif ou non normalise.
                # On le garde comme signal dominant uniquement dans l'ordre relatif
                # des candidats deja preselectionnes.
                score_final=resultat.score_final + score_float,
            )
        )

    resultats_scores.sort(key=lambda resultat: resultat.score_final, reverse=True)
    hors_fenetre = resultats[len(candidats) :]
    return resultats_scores + hors_fenetre


def reranker_resultats(
    question: str,
    candidats: list[tuple[float, Morceau]],
) -> list[ResultatRecherche]:
    """
    Ameliore le classement des resultats trouves par FAISS.

    Technologie suivie :
    - FAISS fournit un score semantique rapide.
    - BM25 fournit un score lexical robuste sur les mots importants.
    - Le score final combine les deux.

    Pourquoi ne pas utiliser seulement FAISS ?
    FAISS peut trouver un passage proche dans le sens, mais parfois un passage
    moins bon sur les mots exacts. BM25 corrige une partie de cela.
    """
    resultats = []
    textes_candidats = [morceau.texte for _, morceau in candidats]
    scores_lexicaux = scores_bm25(question, textes_candidats)

    for (score_semantique, morceau), lexical in zip(candidats, scores_lexicaux):
        score_final = (POIDS_SCORE_SEMANTIQUE * score_semantique) + (
            POIDS_SCORE_BM25 * lexical
        )
        resultats.append(
            ResultatRecherche(
                morceau=morceau,
                score_semantique=score_semantique,
                score_lexical=lexical,
                score_final=score_final,
            )
        )

    return sorted(resultats, key=lambda resultat: resultat.score_final, reverse=True)


def question_demande_valeurs_techniques(question: str) -> bool:
    """Detecte les questions qui demandent des chiffres/specifications."""
    question_minuscule = question.lower()
    indices = [
        "liste",
        "consommation",
        "consommations",
        "electrique",
        "électrique",
        "watts",
        "watt",
        "puissance",
        "spec",
        "specification",
        "spécification",
        "capacite",
        "capacité",
        "capacites",
        "capacités",
        "volumetrie",
        "volumetries",
        "volumétrie",
        "volumétries",
        "capacity",
        "capacities",
        "raw",
        "effective",
        "usable",
        "dimensions",
    ]
    return any(indice in question_minuscule for indice in indices)


def question_demande_portefeuille_capacites(question: str) -> bool:
    """
    Detecte les questions larges qui demandent des capacites sur un portefeuille.

    Dans ce cas, on veut eviter que les 8 meilleurs chunks viennent tous du meme
    document. On prefere une vue plus diversifiee entre plusieurs sources.
    """
    question_minuscule = question.lower()
    mots_portefeuille = ["toutes", "tous", "portfolio", "portefeuille", "solutions"]
    mots_capacite = [
        "volumetrie",
        "volumetries",
        "volumétrie",
        "volumétries",
        "capacite",
        "capacites",
        "capacité",
        "capacités",
        "capacity",
        "capacities",
    ]

    return any(mot in question_minuscule for mot in mots_portefeuille) and any(
        mot in question_minuscule for mot in mots_capacite
    )


def calculer_parametres_recherche_adaptatifs(
    question: str,
    top_k: int,
    candidate_k: int,
    min_score: float,
) -> ParametresRecherche:
    """
    Ajuste automatiquement la largeur de recherche selon la question.

    Principe suivi :
    - top_k = nombre de chunks donnés au modèle final ;
    - candidate_k = nombre de chunks récupérés avant reranking ;
    - min_score = seuil minimal après reranking.

    Pourquoi adapter ?
    Une question très ciblée n'a pas le même besoin qu'une question large.
    Si top_k est trop petit sur une question large, le modèle n'a tout simplement
    pas assez de contexte. S'il est trop grand sur une question simple, on ajoute
    du bruit.
    """
    question_minuscule = question.lower()
    raisons: list[str] = []

    top_k_effectif = top_k
    candidate_k_effectif = max(candidate_k, top_k)
    min_score_effectif = min_score

    if profil_est_actif("everpure") and question_demande_inventaire_produit_everpure(
        question
    ):
        top_k_effectif = max(top_k_effectif, 12)
        candidate_k_effectif = max(candidate_k_effectif, 180)
        min_score_effectif = min(min_score_effectif, 0.0)
        raisons.append("profil everpure : recherche elargie pour inventaire produit")

    elif question_demande_portefeuille_capacites(question):
        top_k_effectif = max(top_k_effectif, 12)
        candidate_k_effectif = max(candidate_k_effectif, 160)
        min_score_effectif = min(min_score_effectif, 0.0)
        raisons.append("question portefeuille/capacités : recherche très élargie")

    elif question_demande_valeurs_techniques(question):
        top_k_effectif = max(top_k_effectif, 8)
        candidate_k_effectif = max(candidate_k_effectif, 120)
        min_score_effectif = min(min_score_effectif, 0.0)
        raisons.append("question technique : recherche élargie et seuil abaissé")

    elif any(
        mot in question_minuscule
        for mot in [
            "liste",
            "lister",
            "toutes",
            "tous",
            "compare",
            "comparaison",
            "synthese",
            "synthèse",
            "avantages",
            "inconvenients",
            "inconvénients",
            "resume",
            "résume",
        ]
    ):
        top_k_effectif = max(top_k_effectif, 10)
        candidate_k_effectif = max(candidate_k_effectif, 120)
        min_score_effectif = min(min_score_effectif, 0.05)
        raisons.append("question de synthèse/liste : contexte élargi")

    else:
        candidate_k_effectif = max(candidate_k_effectif, 60)
        raisons.append("question standard : paramètres utilisateur conservés")

    # Dernière sécurité : le reranking a besoin de plus de candidats que de
    # résultats finaux, sinon il ne peut presque rien améliorer.
    candidate_k_effectif = max(candidate_k_effectif, top_k_effectif * 8)

    return ParametresRecherche(
        top_k=top_k_effectif,
        candidate_k=candidate_k_effectif,
        min_score=min_score_effectif,
        raison="; ".join(raisons),
    )


def filtrer_resultats_pour_valeurs_techniques(
    question: str,
    resultats: list[ResultatRecherche],
) -> list[ResultatRecherche]:
    """
    Wrapper historique qui injecte la detection d'intention technique.

    Le filtrage lui-meme vit dans `rag.retrieval`; la detection metier reste ici
    pour l'instant afin de limiter le deplacement des heuristiques.
    """
    return filtrer_resultats_pour_valeurs_techniques_local(
        question,
        resultats,
        question_demande_valeurs_techniques_fn=question_demande_valeurs_techniques,
    )



def rechercher(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    min_score: float = DEFAULT_MIN_SCORE,
    filtre: FiltreMetadata | None = None,
    adaptatif: bool = True,
    utiliser_query_rewrite_llm: bool | None = None,
    utiliser_reranker_llm: bool | None = None,
    utiliser_cross_encoder_reranker: bool | None = None,
) -> list[ResultatRecherche]:
    """
    Recherche les morceaux les plus pertinents pour une question.
    """
    charger_env_local()

    if adaptatif:
        parametres = calculer_parametres_recherche_adaptatifs(
            question=question,
            top_k=top_k,
            candidate_k=candidate_k,
            min_score=min_score,
        )
        top_k = parametres.top_k
        candidate_k = parametres.candidate_k
        min_score = parametres.min_score

    index, morceaux = charger_index()
    filtre = filtre or FiltreMetadata()

    # On construit plusieurs requetes de retrieval :
    # - la question originale ;
    # - une reformulation LLM courte ;
    # - une expansion deterministe avec synonymes et acronymes.
    #
    # Chaque requete interroge FAISS separement. On fusionne ensuite les chunks.
    # C'est plus propre qu'une seule requete geante, car chaque strategie garde
    # son signal propre.
    requetes_recherche = construire_requetes_recherche(
        question,
        utiliser_query_rewrite_llm=utiliser_query_rewrite_llm,
    )
    textes_requetes = [requete for _, requete in requetes_recherche]
    vecteurs_questions = creer_embeddings(textes_requetes)
    question_recherche = "\n\n".join(
        f"{libelle} : {requete}" for libelle, requete in requetes_recherche
    )

    # On recupere plus de candidats que le nombre final voulu.
    # C'est indispensable pour permettre le reranking et les filtres metadata.
    nombre_total = index.ntotal
    nombre_candidats = min(nombre_total, max(top_k, candidate_k))

    if (
        filtre.fichiers
        or filtre.page_min is not None
        or filtre.page_max is not None
        or filtre.source_types
    ):
        nombre_candidats = nombre_total

    resultats_par_requete: list[tuple[str, list[tuple[float, int]]]] = []

    for (libelle, _), vecteur_question in zip(requetes_recherche, vecteurs_questions):
        scores, indices = index.search(vecteur_question.reshape(1, -1), nombre_candidats)
        resultats_par_requete.append((libelle, list(zip(scores[0], indices[0]))))

    candidats = fusionner_candidats_multi_requetes(
        resultats_par_requete=resultats_par_requete,
        morceaux=morceaux,
        filtre=filtre,
    )

    tous_les_resultats_rerankes = reranker_resultats(question_recherche, candidats)
    resultats_rerankes = [
        resultat for resultat in tous_les_resultats_rerankes if resultat.score_final >= min_score
    ]
    resultats_rerankes = filtrer_resultats_pour_valeurs_techniques(question, resultats_rerankes)

    if profil_est_actif("everpure"):
        resultats_rerankes = prioriser_inventaires_produits_everpure(
            question,
            resultats_rerankes,
        )

    if utiliser_cross_encoder_reranker is None:
        utiliser_cross_encoder_reranker = variable_env_booleenne(
            "RAG_CROSS_ENCODER_RERANKER",
            DEFAULT_CROSS_ENCODER_RERANKER,
        )

    if utiliser_cross_encoder_reranker:
        resultats_rerankes = reranker_resultats_avec_cross_encoder(
            question=question,
            resultats=resultats_rerankes,
            top_k=top_k,
        )

    if utiliser_reranker_llm is None:
        utiliser_reranker_llm = variable_env_booleenne(
            "RAG_LLM_RERANKER",
            DEFAULT_LLM_RERANKER,
        )

    if utiliser_reranker_llm:
        resultats_rerankes = reranker_resultats_avec_llm(
            question=question,
            resultats=resultats_rerankes,
            top_k=top_k,
        )

    if question_demande_portefeuille_capacites(question):
        resultats_rerankes = diversifier_resultats_par_document(
            resultats_rerankes,
            top_k=top_k,
            max_par_document=2,
        )

    return resultats_rerankes[:top_k]


# Pourquoi cette etape ?
# -> On ne donne pas tous les PDF au modele.
# -> On lui donne seulement les passages les plus proches semantiquement.


# Le prompt et le contexte long vivent dans `rag.prompting`.
# On les importe plus haut pour garder l'API historique de `rag_pdf.py`.


# ==============================
# --- 12. Pipeline complet ---
# ==============================

def demander(
    question: str,
    top_k: int | None = None,
    candidate_k: int | None = None,
    budget_tokens: int | None = None,
    min_score: float | None = None,
    filtre: FiltreMetadata | None = None,
    modele_generation: str | None = None,
    utiliser_query_rewrite_llm: bool | None = None,
    utiliser_reranker_llm: bool | None = None,
    utiliser_cross_encoder_reranker: bool | None = None,
) -> str:
    """
    Pipeline complet :
    question -> recherche FAISS -> prompt -> reponse OpenAI.
    """
    charger_env_local()

    top_k = top_k or int(os.getenv("TOP_K", str(DEFAULT_TOP_K)))
    candidate_k = candidate_k or int(os.getenv("CANDIDATE_K", str(DEFAULT_CANDIDATE_K)))
    budget_tokens = budget_tokens or int(
        os.getenv("CONTEXT_TOKEN_BUDGET", str(DEFAULT_CONTEXT_TOKEN_BUDGET))
    )
    min_score = min_score if min_score is not None else float(
        os.getenv("MIN_SCORE", str(DEFAULT_MIN_SCORE))
    )

    resultats = rechercher(
        question,
        top_k=top_k,
        candidate_k=candidate_k,
        min_score=min_score,
        filtre=filtre,
        utiliser_query_rewrite_llm=utiliser_query_rewrite_llm,
        utiliser_reranker_llm=utiliser_reranker_llm,
        utiliser_cross_encoder_reranker=utiliser_cross_encoder_reranker,
    )
    prompt = construire_prompt(question, resultats, budget_tokens=budget_tokens)
    reponse = appeler_modele(prompt, modele=modele_generation)

    return reponse


def afficher_sources(
    question: str,
    top_k: int | None = None,
    candidate_k: int | None = None,
    min_score: float | None = None,
    filtre: FiltreMetadata | None = None,
    utiliser_query_rewrite_llm: bool | None = None,
    utiliser_reranker_llm: bool | None = None,
    utiliser_cross_encoder_reranker: bool | None = None,
) -> None:
    """Affiche les sources retrouvees pour mieux comprendre le RAG."""
    top_k = top_k or int(os.getenv("TOP_K", str(DEFAULT_TOP_K)))
    candidate_k = candidate_k or int(os.getenv("CANDIDATE_K", str(DEFAULT_CANDIDATE_K)))
    min_score = min_score if min_score is not None else float(
        os.getenv("MIN_SCORE", str(DEFAULT_MIN_SCORE))
    )
    resultats = rechercher(
        question,
        top_k=top_k,
        candidate_k=candidate_k,
        min_score=min_score,
        filtre=filtre,
        utiliser_query_rewrite_llm=utiliser_query_rewrite_llm,
        utiliser_reranker_llm=utiliser_reranker_llm,
        utiliser_cross_encoder_reranker=utiliser_cross_encoder_reranker,
    )

    print("\nSources retrouvees")
    print("==================")

    if not resultats:
        print(f"Aucun resultat au-dessus du seuil MIN_SCORE={min_score:.3f}.")
        return

    for numero, resultat in enumerate(resultats, start=1):
        morceau = resultat.morceau
        if getattr(morceau, "source_type", "pdf") == "web":
            localisation = (
                f"Web : {morceau.fichier} | {getattr(morceau, 'url', '')} | "
                f"consulte le {getattr(morceau, 'date_consultation', '')}"
            )
        else:
            localisation = f"{morceau.fichier}, page {morceau.page}"

        print(
            f"{numero}. {localisation}, chunk {getattr(morceau, 'numero', '?')} | "
            f"final {resultat.score_final:.3f} | "
            f"FAISS {resultat.score_semantique:.3f} | "
            f"lexical {resultat.score_lexical:.3f}"
        )


# ==============================
# --- 13. CLI ---
# ==============================

def main() -> None:
    """
    Interface en ligne de commande.

    Exemples :

    python rag_pdf.py indexer
    python rag_pdf.py demander "Quels sont les points importants ?"
    """
    parseur = argparse.ArgumentParser(
        description="RAG documentaire avec OpenAI, FAISS et sources citees."
    )
    sous_commandes = parseur.add_subparsers(dest="commande", required=True)

    sous_commandes.add_parser(
        "indexer",
        help="Lire les PDF + sources web locales et construire l'index FAISS.",
    )

    commande_web = sous_commandes.add_parser(
        "web-indexer",
        help="Telecharger des pages web autorisees puis reconstruire l'index FAISS.",
    )
    commande_web.add_argument("urls", nargs="+", help="URLs web a indexer.")
    commande_web.add_argument(
        "--no-rebuild",
        action="store_true",
        help="Sauvegarder les pages web sans reconstruire FAISS.",
    )

    commande_demander = sous_commandes.add_parser(
        "demander",
        help="Poser une question aux PDF indexes.",
    )
    commande_demander.add_argument("question", help="Question a poser au RAG.")
    commande_demander.add_argument(
        "--sources",
        action="store_true",
        help="Afficher aussi les sources retrouvees par FAISS.",
    )
    commande_demander.add_argument(
        "--top-k",
        "--top_k",
        dest="top_k",
        type=int,
        default=None,
        help="Nombre de morceaux gardes apres reranking.",
    )
    commande_demander.add_argument(
        "--candidate-k",
        type=int,
        default=None,
        help="Nombre de candidats FAISS avant reranking.",
    )
    commande_demander.add_argument(
        "--context-tokens",
        type=int,
        default=None,
        help="Budget approximatif de tokens pour le contexte.",
    )
    commande_demander.add_argument(
        "--min-score",
        "--min_score",
        dest="min_score",
        type=float,
        default=None,
        help="Seuil minimal de score final apres reranking.",
    )
    commande_demander.add_argument(
        "--document",
        action="append",
        default=None,
        help="Filtrer sur un nom de PDF. Option repetable.",
    )
    commande_demander.add_argument(
        "--page-min",
        type=int,
        default=None,
        help="Filtrer a partir de cette page.",
    )
    commande_demander.add_argument(
        "--page-max",
        type=int,
        default=None,
        help="Filtrer jusqu'a cette page.",
    )
    commande_demander.add_argument(
        "--source-type",
        choices=["pdf", "web"],
        action="append",
        default=None,
        help="Limiter la recherche a un type de source. Option repetable.",
    )
    commande_demander.add_argument(
        "--no-query-rewrite",
        action="store_true",
        help="Desactiver la reformulation LLM pour cette question.",
    )
    commande_demander.add_argument(
        "--no-llm-reranker",
        action="store_true",
        help="Desactiver le reranking LLM pour cette question.",
    )
    commande_demander.add_argument(
        "--cross-encoder-reranker",
        action="store_true",
        help=(
            "Activer le reranker local CrossEncoder. "
            "Necessite sentence-transformers et peut telecharger le modele au premier usage."
        ),
    )

    arguments = parseur.parse_args()

    if arguments.commande == "indexer":
        indexer_pdf()
        return

    if arguments.commande == "web-indexer":
        indexer_pages_web(arguments.urls)
        if not arguments.no_rebuild:
            indexer_pdf()
        return

    if arguments.commande == "demander":
        filtre = FiltreMetadata(
            fichiers=set(arguments.document) if arguments.document else None,
            page_min=arguments.page_min,
            page_max=arguments.page_max,
            source_types=set(arguments.source_type) if arguments.source_type else None,
        )
        reponse = demander(
            arguments.question,
            top_k=arguments.top_k,
            candidate_k=arguments.candidate_k,
            budget_tokens=arguments.context_tokens,
            min_score=arguments.min_score,
            filtre=filtre,
            utiliser_query_rewrite_llm=not arguments.no_query_rewrite,
            utiliser_reranker_llm=not arguments.no_llm_reranker,
            utiliser_cross_encoder_reranker=arguments.cross_encoder_reranker,
        )

        print("\nReponse")
        print("=======")
        print(reponse)

        if arguments.sources:
            afficher_sources(
                arguments.question,
                top_k=arguments.top_k,
                candidate_k=arguments.candidate_k,
                min_score=arguments.min_score,
                filtre=filtre,
                utiliser_query_rewrite_llm=not arguments.no_query_rewrite,
                utiliser_reranker_llm=not arguments.no_llm_reranker,
                utiliser_cross_encoder_reranker=arguments.cross_encoder_reranker,
            )


if __name__ == "__main__":
    main()


# Pourquoi une CLI ?
# -> On separe l'indexation et les questions.
# -> On indexe rarement, mais on pose souvent des questions.
# -> C'est un fonctionnement standard dans les petits pipelines RAG.
