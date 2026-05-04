"""
Reranking des resultats.

Responsabilite :
ameliorer l'ordre des chunks apres FAISS avec BM25, reranker LLM optionnel et
reranker local BGE optionnel.

Pourquoi isoler ce module ?
- Le reranking est une brique explicable en demo Sales Engineering.
- Il melange cache, LLM distant et worker local : mieux vaut le separer du
  moteur d'orchestration.
- `rag.engine` peut rester lisible : recherche, filtres, prompt, generation.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import pickle
import re
import select
import subprocess
from pathlib import Path

from rag.config import (
    BASE_DIR,
    INDEX_DIR,
    LLM_RERANK_CACHE_FILE,
    DEFAULT_CHAT_MODEL,
    DEFAULT_LLM_RERANK_CANDIDATES,
    DEFAULT_CROSS_ENCODER_MODEL,
    DEFAULT_CROSS_ENCODER_RERANK_CANDIDATES,
    DEFAULT_BGE_WORKER_TIMEOUT_SECONDS,
    LLM_RERANK_PROMPT_VERSION,
    POIDS_SCORE_SEMANTIQUE,
    POIDS_SCORE_BM25,
)
from rag.llm import creer_client_openai
from rag.models import Morceau, ResultatRecherche
from rag.retrieval import scores_bm25, variable_env_booleenne


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



# Compatibilite historique : ces helpers specifiques au profil Everpure etaient
# temporairement re-exportes depuis `rag.reranking`. On les garde pour ne pas
# casser un ancien import externe.
from rag.everpure import (  # noqa: E402,F401
    nombre_modeles_flasharray,
    prioriser_inventaire_flasharray,
    nombre_modeles_flashblade,
    prioriser_inventaire_flashblade,
)
