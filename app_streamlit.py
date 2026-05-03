"""
Interface Streamlit pour le mini RAG PDF.

Objectif :
- garder le moteur RAG dans rag_pdf.py ;
- ajouter une interface simple pour charger des PDF, indexer, poser une question ;
- afficher les sources et les scores pour comprendre le retrieval.

Streamlit est choisi parce que c'est tres rapide pour un POC :
une seule commande lance une interface web locale.
"""

from __future__ import annotations

import contextlib
import io
import os
import re
import shutil
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import streamlit as st
from dotenv import load_dotenv

import rag_pdf


load_dotenv()

STATIC_PDF_DIR = Path(__file__).resolve().parent / "static" / "pdfs"


st.set_page_config(
    page_title="Everpure RAG",
    page_icon="◈",
    layout="wide",
)


def signature_index() -> tuple[float, float]:
    """
    Signature légère de l'index.

    Streamlit réexécute ce fichier à chaque interaction. Pour éviter de relire
    FAISS et `morceaux.pkl` simplement pour afficher la page, on met certaines
    données en cache. Cette signature invalide le cache dès que l'index change.
    """
    faiss_mtime = rag_pdf.FAISS_INDEX_FILE.stat().st_mtime if rag_pdf.FAISS_INDEX_FILE.exists() else 0
    metadata_mtime = rag_pdf.METADATA_FILE.stat().st_mtime if rag_pdf.METADATA_FILE.exists() else 0
    return faiss_mtime, metadata_mtime


@st.cache_data(show_spinner=False)
def documents_indexes_cached(_signature: tuple[float, float]) -> list[str]:
    """Liste des documents indexés, cachée pour accélérer l'affichage."""
    return rag_pdf.lister_documents_indexes()


@st.cache_data(show_spinner=False)
def statistiques_index_cached(_signature: tuple[float, float]) -> dict[str, int | bool]:
    """Statistiques d'index, cachées pour les panneaux Settings."""
    return rag_pdf.statistiques_index()

st.markdown(
    """
    <style>
    html, body, .stApp {
        color-scheme: light !important;
    }

    :root {
        --everpure-ink: #27231f;
        --everpure-charcoal: #1f1d1a;
        --everpure-orange: #de5a1b;
        --everpure-burnt: #bd643c;
        --everpure-cream: #f8efdc;
        --everpure-surface: #fff8ec;
        --everpure-white: #ffffff;
        --everpure-peach: #f6c8b5;
        --everpure-line: #ead8bd;
    }

    .stApp {
        background: #fbf2df !important;
        background-image: none !important;
        color: var(--everpure-ink) !important;
    }

    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stMainBlockContainer"],
    [data-testid="stDecoration"],
    section.main {
        background: #fbf2df !important;
        background-image: none !important;
    }

    [data-testid="stToolbar"],
    [data-testid="stStatusWidget"],
    [data-testid="stMainMenu"],
    footer {
        visibility: hidden !important;
        height: 0 !important;
    }

    h1, h2, h3 {
        color: var(--everpure-ink);
        letter-spacing: 0;
    }

    [data-testid="stHeader"] {
        background: #fbf2df !important;
        border-bottom: 1px solid var(--everpure-line) !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--everpure-line) !important;
        background: rgba(255, 250, 240, 0.96) !important;
        box-shadow: 0 10px 24px rgba(31, 29, 26, 0.08) !important;
    }

    .everpure-hero {
        border: 1px solid rgba(222, 90, 27, 0.28);
        border-radius: 8px;
        padding: 18px 22px;
        margin: 4px 0 12px 0;
        background: var(--everpure-surface);
        color: var(--everpure-charcoal);
        box-shadow: none;
    }

    .everpure-hero h1 {
        color: var(--everpure-charcoal);
        margin: 0;
        font-size: 2.4rem;
        line-height: 1.05;
        font-weight: 760;
    }

    .everpure-hero p {
        color: var(--everpure-charcoal);
        margin: 10px 0 0 0;
        font-size: 1rem;
    }

    .everpure-kicker {
        display: inline-block;
        color: var(--everpure-orange);
        font-size: 0.78rem;
        text-transform: uppercase;
        font-weight: 700;
        margin-bottom: 8px;
    }

    [data-testid="stExpander"] {
        background: var(--everpure-white) !important;
        border: 1px solid var(--everpure-line) !important;
        border-radius: 8px !important;
    }

    [data-testid="stExpander"] details,
    [data-testid="stExpander"] summary {
        background: var(--everpure-white) !important;
        color: var(--everpure-charcoal) !important;
    }

    [data-testid="stMetric"] {
        background: linear-gradient(180deg, rgba(251, 242, 223, 0.94), rgba(255, 255, 255, 0.86));
        border: 1px solid var(--everpure-line);
        border-radius: 12px;
        padding: 12px;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid var(--everpure-line);
        border-radius: 12px;
        overflow: hidden;
    }

    .rag-answer {
        max-width: 100%;
        overflow-x: auto;
    }

    .rag-answer table {
        display: block;
        width: 100%;
        max-width: 100%;
        overflow-x: auto;
        border-collapse: collapse;
        table-layout: auto;
        font-size: 0.92rem;
    }

    .rag-answer th,
    .rag-answer td {
        min-width: 140px;
        max-width: 280px;
        white-space: normal !important;
        overflow-wrap: anywhere;
        word-break: normal;
        vertical-align: top;
        padding: 0.7rem 0.8rem;
    }

    .rag-answer th:first-child,
    .rag-answer td:first-child,
    .rag-answer th:last-child,
    .rag-answer td:last-child {
        min-width: 90px;
        max-width: 150px;
    }

    textarea,
    input,
    select,
    div[role="textbox"],
    [data-baseweb="textarea"],
    [data-baseweb="input"],
    [data-baseweb="select"],
    [data-baseweb="base-input"],
    [data-baseweb="textarea"] > div,
    [data-baseweb="input"] > div,
    [data-baseweb="select"] > div,
    [data-baseweb="base-input"] > div {
        border-color: var(--everpure-line) !important;
        background: var(--everpure-white) !important;
        color: var(--everpure-charcoal) !important;
    }

    [data-baseweb="textarea"] textarea,
    [data-baseweb="input"] input,
    [data-baseweb="base-input"] input,
    [data-baseweb="select"] div {
        background: var(--everpure-white) !important;
        color: var(--everpure-charcoal) !important;
        -webkit-text-fill-color: var(--everpure-charcoal) !important;
    }

    [data-baseweb="textarea"] textarea:focus,
    [data-baseweb="input"] input:focus {
        border-color: var(--everpure-orange) !important;
        box-shadow: 0 0 0 1px rgba(222, 90, 27, 0.22) !important;
    }

    textarea::placeholder,
    input::placeholder {
        color: #8f8375 !important;
        opacity: 1 !important;
    }

    [data-testid="stMarkdownContainer"] code {
        background: #fff3e8;
        color: var(--everpure-charcoal);
    }

    .stButton > button,
    .stDownloadButton > button {
        border-color: var(--everpure-orange) !important;
        color: var(--everpure-ink) !important;
        background: var(--everpure-white) !important;
        border-radius: 8px !important;
        box-shadow: none !important;
    }

    div[data-testid="stButton"] {
        width: auto !important;
    }

    .stButton > button[kind="primary"],
    .stDownloadButton > button[kind="primary"] {
        background: var(--everpure-orange) !important;
        color: #ffffff !important;
        border-color: var(--everpure-orange) !important;
    }

    a {
        color: var(--everpure-orange);
        font-weight: 650;
    }

    div[data-testid="stTextArea"] label,
    div[data-testid="stTextInput"] label {
        color: var(--everpure-charcoal) !important;
    }

    button[kind="secondary"] {
        background: var(--everpure-white) !important;
        color: var(--everpure-charcoal) !important;
    }

    .st-key-settings_button {
        display: flex !important;
        justify-content: flex-end !important;
    }

    .st-key-settings_button button {
        min-height: 44px !important;
        height: 44px !important;
        width: 48px !important;
        padding: 0 !important;
        border-radius: 8px !important;
        background: var(--everpure-orange) !important;
        color: #ffffff !important;
        border: 1px solid var(--everpure-orange) !important;
        box-shadow: none;
        font-size: 1.05rem !important;
        line-height: 1 !important;
    }

    .st-key-settings_button button p {
        color: #ffffff !important;
        margin: 0 !important;
        line-height: 1 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def cle_openai_presente() -> bool:
    """Verifie seulement si la cle existe, sans jamais l'afficher."""
    cle = os.getenv("OPENAI_API_KEY")
    return bool(cle and cle != "sk-...")


def initialiser_etat_session() -> None:
    """Initialise tous les réglages persistés côté Streamlit."""
    valeurs_defaut = {
        "source_types": ["pdf", "web"],
        "documents_selectionnes": [],
        "page_min": 0,
        "page_max": 0,
        "mode_debug": False,
        "historique_questions": [],
        "afficher_settings": False,
        "dernier_export_markdown": "",
        "modele_generation": os.getenv("OPENAI_MODEL", rag_pdf.DEFAULT_CHAT_MODEL),
        "top_k": int(os.getenv("TOP_K", rag_pdf.DEFAULT_TOP_K)),
        "candidate_k": int(os.getenv("CANDIDATE_K", rag_pdf.DEFAULT_CANDIDATE_K)),
        "min_score": float(os.getenv("MIN_SCORE", rag_pdf.DEFAULT_MIN_SCORE)),
        "context_tokens": int(os.getenv("CONTEXT_TOKEN_BUDGET", rag_pdf.DEFAULT_CONTEXT_TOKEN_BUDGET)),
        "query_rewrite_llm": rag_pdf.variable_env_booleenne(
            "RAG_QUERY_REWRITE_LLM",
            rag_pdf.DEFAULT_QUERY_REWRITE_LLM,
        ),
        "llm_reranker": rag_pdf.variable_env_booleenne(
            "RAG_LLM_RERANKER",
            rag_pdf.DEFAULT_LLM_RERANKER,
        ),
        "cross_encoder_reranker": rag_pdf.variable_env_booleenne(
            "RAG_CROSS_ENCODER_RERANKER",
            rag_pdf.DEFAULT_CROSS_ENCODER_RERANKER,
        ),
        "comparer_reranker": False,
    }

    for cle, valeur in valeurs_defaut.items():
        if cle not in st.session_state:
            st.session_state[cle] = valeur

    st.session_state.candidate_k = max(st.session_state.candidate_k, st.session_state.top_k)


def sauver_pdfs(uploaded_files) -> list[Path]:
    """
    Sauvegarde les PDF deposes dans l'interface.

    Les fichiers sont stockes dans data/pdfs, exactement comme pour la CLI.
    Il faudra ensuite relancer l'indexation pour les prendre en compte.
    """
    rag_pdf.PDF_DIR.mkdir(parents=True, exist_ok=True)
    chemins = []

    for fichier in uploaded_files:
        chemin = rag_pdf.PDF_DIR / fichier.name
        chemin.write_bytes(fichier.getbuffer())
        chemins.append(chemin)

    synchroniser_pdfs_statiques()
    return chemins


def synchroniser_pdfs_statiques() -> None:
    """
    Rend les PDF accessibles depuis le navigateur Streamlit.

    Pourquoi ?
    Les PDF sources vivent dans data/pdfs, mais une page web ne peut pas ouvrir
    librement un fichier local du disque. Streamlit peut en revanche servir les
    fichiers places dans le dossier static si l'option static serving est active.

    On cree donc un miroir local dans static/pdfs.
    - hardlink si possible : pas de duplication reelle sur le disque ;
    - copie classique en secours si le systeme de fichiers refuse le hardlink.
    """
    STATIC_PDF_DIR.mkdir(parents=True, exist_ok=True)

    for pdf in rag_pdf.lister_pdfs():
        destination = STATIC_PDF_DIR / pdf.name

        if destination.exists() and destination.stat().st_size == pdf.stat().st_size:
            continue

        if destination.exists():
            destination.unlink()

        try:
            destination.hardlink_to(pdf)
        except OSError:
            shutil.copy2(pdf, destination)


def lien_source_pdf(nom_fichier: str, page: int) -> str | None:
    """
    Construit une URL cliquable vers le PDF et, si le navigateur le supporte,
    directement vers la page citee.
    """
    chemin_pdf = rag_pdf.PDF_DIR / nom_fichier

    if not chemin_pdf.exists():
        return None

    synchroniser_pdfs_statiques()
    nom_encode = quote(nom_fichier)
    return f"/app/static/pdfs/{nom_encode}#page={page}"


def url_source(morceau: rag_pdf.Morceau) -> str | None:
    """Retourne l'URL cliquable d'une source, web ou PDF local."""
    if getattr(morceau, "source_type", "pdf") == "web":
        return getattr(morceau, "url", "") or None

    return lien_source_pdf(morceau.fichier, morceau.page)


def libelle_source(morceau: rag_pdf.Morceau) -> str:
    """Libelle court affiche au survol du lien source."""
    if getattr(morceau, "source_type", "pdf") == "web":
        return f"Source web : {morceau.fichier}"

    return f"{morceau.fichier}, page {morceau.page}"


def construire_references_sources(
    resultats: list[rag_pdf.ResultatRecherche],
) -> list[dict[str, str | int]]:
    """
    Construit la liste des sources avec les memes numeros que le contexte RAG.

    Important :
    le modele cite [1], [2], etc. en fonction de l'ordre des chunks transmis.
    On ne deduplique donc pas ici, meme si deux chunks viennent du meme PDF et
    de la meme page, sinon certains numeros cites ne seraient plus cliquables.
    """
    references: list[dict[str, str | int]] = []

    for index, resultat in enumerate(resultats, start=1):
        morceau = resultat.morceau
        url = url_source(morceau)

        if not url:
            continue

        references.append(
            {
                "numero": index,
                "url": url,
                "libelle": libelle_source(morceau),
                "fichier": morceau.fichier,
                "page": morceau.page,
                "source_type": getattr(morceau, "source_type", "pdf"),
            }
        )

    return references


def lien_reference(reference: dict[str, str | int]) -> str:
    """Retourne un lien Markdown numerote, stable dans les tableaux."""
    numero = reference["numero"]
    url = str(reference["url"]).replace(")", "%29").replace(" ", "%20")
    return f"[[{numero}]]({url})"


def remplacer_citations_par_liens(
    reponse: str,
    references: list[dict[str, str | int]],
) -> str:
    """
    Remplace les citations longues produites par le modele par des numeros.

    Exemple :
    "(ds-flasharray-xl.pdf, page 4)" devient "[1]" cliquable.
    """
    references_par_numero = {
        int(reference["numero"]): reference for reference in references
    }

    def remplacer_numero(match: re.Match) -> str:
        numero = int(match.group(1))
        reference = references_par_numero.get(numero)
        if not reference:
            return match.group(0)
        return lien_reference(reference)

    # Une seule passe evite les liens HTML imbriques dans les tableaux Markdown.
    reponse_html = re.sub(r"\[(\d+)\]", remplacer_numero, reponse)

    for reference in references:
        lien = lien_reference(reference)

        if reference["source_type"] == "web":
            url = str(reference["url"])
            if url:
                reponse_html = reponse_html.replace(url, lien)
            continue

        fichier = str(reference["fichier"])
        page = int(reference["page"])
        fichier_regex = re.escape(fichier)
        motifs = [
            rf"\({fichier_regex}\s*,\s*page\s*{page}\)",
            rf"{fichier_regex}\s*,\s*page\s*{page}",
            rf"\({fichier_regex}\s*,\s*p\.\s*{page}\)",
            rf"{fichier_regex}\s*,\s*p\.\s*{page}",
        ]

        for motif in motifs:
            reponse_html = re.sub(motif, lien, reponse_html, flags=re.IGNORECASE)

    if references and "](" not in reponse_html:
        liens = " ".join(lien_reference(reference) for reference in references[:3])
        reponse_html = f"{reponse_html}\n\n{liens}"

    return reponse_html


def generer_reponse_avec_sources(
    reponse: str,
    resultats: list[rag_pdf.ResultatRecherche],
) -> str:
    """Retourne la reponse avec les references source integrees dans le texte."""
    references = construire_references_sources(resultats)
    return remplacer_citations_par_liens(reponse, references)


def afficher_reponse_avec_sources(
    reponse: str,
    resultats: list[rag_pdf.ResultatRecherche],
) -> str:
    """Affiche la reponse avec les references source integrees dans le texte."""
    reponse_markdown = generer_reponse_avec_sources(reponse, resultats)
    st.markdown('<div class="rag-answer">', unsafe_allow_html=True)
    st.markdown(reponse_markdown)
    st.markdown("</div>", unsafe_allow_html=True)
    return reponse_markdown


def afficher_debug_sources(resultats: list[rag_pdf.ResultatRecherche]) -> None:
    """Affiche les details retrieval uniquement quand l'utilisateur les ouvre."""
    with st.expander("Debug sources : extraits et scores", expanded=False):
        for index, resultat in enumerate(resultats, start=1):
            morceau = resultat.morceau
            st.markdown(f"**[{index}] {libelle_source(morceau)}**")
            st.write(morceau.texte)
            if getattr(morceau, "source_type", "pdf") == "web":
                st.caption(
                    f"URL : {getattr(morceau, 'url', '')} | "
                    f"Consulté le : {getattr(morceau, 'date_consultation', '')}"
                )
            st.caption(
                f"Score FAISS : {resultat.score_semantique:.3f} | "
                f"Score BM25 : {resultat.score_lexical:.3f} | "
                f"Score final : {resultat.score_final:.3f}"
            )


def afficher_liste_sources_compacte(resultats: list[rag_pdf.ResultatRecherche]) -> None:
    """Affiche les sources sous forme compacte pour comparer deux rankings."""
    if not resultats:
        st.info("Aucun résultat.")
        return

    for index, resultat in enumerate(resultats, start=1):
        morceau = resultat.morceau
        st.markdown(f"**{index}. {libelle_source(morceau)}**")
        st.caption(
            f"Score final {resultat.score_final:.3f} | "
            f"FAISS {resultat.score_semantique:.3f} | "
            f"BM25 {resultat.score_lexical:.3f}"
        )


def afficher_comparaison_reranker(
    resultats_sans: list[rag_pdf.ResultatRecherche],
    resultats_avec: list[rag_pdf.ResultatRecherche],
) -> None:
    """
    Montre l'impact du reranker LLM.

    Lecture :
    - Sans reranker : ordre FAISS + BM25 + règles métier.
    - Avec reranker : le LLM reclasse les meilleurs candidats selon leur utilité
      pour répondre à la question.
    """
    with st.expander("Comparaison retrieval : base / rerankers avancés", expanded=True):
        col_sans, col_avec = st.columns(2)

        with col_sans:
            st.markdown("**Base : FAISS + BM25**")
            afficher_liste_sources_compacte(resultats_sans)

        with col_avec:
            st.markdown("**Avec rerankers sélectionnés**")
            afficher_liste_sources_compacte(resultats_avec)

        top_sans = [r.morceau.fichier + str(r.morceau.page) + str(r.morceau.numero) for r in resultats_sans[:5]]
        top_avec = [r.morceau.fichier + str(r.morceau.page) + str(r.morceau.numero) for r in resultats_avec[:5]]

        if top_sans == top_avec:
            st.caption("Top 5 identique : les rerankers confirment le classement initial.")
        else:
            st.caption("Top 5 différent : les rerankers ont modifié l'ordre ou la sélection des sources.")


def afficher_debug_recherche(
    question: str,
    resultats: list[rag_pdf.ResultatRecherche],
    top_k_effectif: int,
    candidate_k_effectif: int,
    min_score_effectif: float,
    context_tokens: int,
    raison_adaptation: str,
    query_rewrite_llm: bool,
    llm_reranker: bool,
    cross_encoder_reranker: bool,
) -> None:
    """Affiche un diagnostic lisible du retrieval pour comprendre une réponse."""
    with st.expander("Debug retrieval : requête, chunks et scores", expanded=False):
        requetes_recherche = rag_pdf.construire_requetes_recherche(
            question,
            utiliser_query_rewrite_llm=query_rewrite_llm,
        )

        st.markdown("**Question utilisateur**")
        st.code(question, language="text")

        st.markdown("**Requêtes utilisées pour la recherche**")
        for libelle, requete in requetes_recherche:
            st.caption(libelle)
            st.code(requete, language="text")

        st.markdown("**Paramètres effectifs**")
        st.write(
            {
                "top_k": top_k_effectif,
                "candidate_k": candidate_k_effectif,
                "min_score": min_score_effectif,
                "context_tokens": context_tokens,
                "adaptation": raison_adaptation,
                "query_rewrite_llm": query_rewrite_llm,
                "llm_reranker": llm_reranker,
                "cross_encoder_reranker": cross_encoder_reranker,
                "cross_encoder_installe": rag_pdf.cross_encoder_disponible(),
                "chunks_retrouves": len(resultats),
            }
        )

        if not resultats:
            st.info("Aucun chunk récupéré avec ces paramètres.")
            return

        st.markdown("**Chunks récupérés**")
        for index, resultat in enumerate(resultats, start=1):
            morceau = resultat.morceau
            st.markdown(f"**[{index}] {libelle_source(morceau)}**")
            st.caption(
                f"Score final {resultat.score_final:.3f} | "
                f"FAISS {resultat.score_semantique:.3f} | "
                f"BM25 {resultat.score_lexical:.3f} | "
                f"chunk {getattr(morceau, 'numero', '?')}"
            )
            st.text(morceau.texte[:1400])


def afficher_trace_rag(
    trace: dict,
    resultats: list[rag_pdf.ResultatRecherche],
    prompt: str,
) -> None:
    """
    Affiche une trace RAG lisible pour expliquer comment la réponse est née.

    Objectif pédagogique :
    un bon RAG ne doit pas seulement répondre ; il doit permettre de comprendre
    pourquoi il a répondu ainsi. Cette trace montre donc le chemin complet :
    question -> retrieval -> reranking -> contexte -> génération.
    """
    with st.expander("Trace RAG : comment la réponse a été construite", expanded=False):
        st.markdown("**Résumé d'exécution**")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Retrieval", f"{trace['temps_retrieval_s']:.2f} s")
        col2.metric("Génération", f"{trace['temps_generation_s']:.2f} s")
        col3.metric("Total", f"{trace['temps_total_s']:.2f} s")
        col4.metric("Chunks transmis", len(resultats))

        st.markdown("**Paramètres utilisés**")
        st.json(
            {
                "modele_generation": trace["modele_generation"],
                "top_k_effectif": trace["top_k_effectif"],
                "candidate_k_effectif": trace["candidate_k_effectif"],
                "min_score_effectif": trace["min_score_effectif"],
                "budget_contexte": trace["context_tokens"],
                "query_rewrite_llm": trace["query_rewrite_llm"],
                "reranker_llm": trace["llm_reranker"],
                "bge_demande": trace["cross_encoder_reranker"],
                "bge_reellement_actif": trace["bge_actif"],
                "adaptation": trace["raison_adaptation"],
                "fallback_utilise": trace["fallback_utilise"],
            }
        )

        st.markdown("**Requêtes de recherche**")
        for libelle, requete in trace["requetes_recherche"]:
            st.caption(libelle)
            st.code(requete, language="text")

        if resultats:
            st.markdown("**Sources retenues pour le prompt**")
            lignes_sources = []
            for index, resultat in enumerate(resultats, start=1):
                morceau = resultat.morceau
                lignes_sources.append(
                    {
                        "N": index,
                        "Document": morceau.fichier,
                        "Page": morceau.page,
                        "Titre": getattr(morceau, "titre", "") or "",
                        "FAISS": round(resultat.score_semantique, 3),
                        "BM25": round(resultat.score_lexical, 3),
                        "Final": round(resultat.score_final, 3),
                    }
                )
            st.dataframe(lignes_sources, use_container_width=True, hide_index=True)

            st.markdown("**Extraits transmis**")
            for index, resultat in enumerate(resultats, start=1):
                morceau = resultat.morceau
                st.markdown(f"**[{index}] {libelle_source(morceau)}**")
                st.text(" ".join(morceau.texte.split())[:1200])
        else:
            st.info("Aucun chunk n'a été transmis au modèle.")

        if st.checkbox("Afficher le prompt complet", value=False, key="afficher_prompt_trace"):
            st.code(prompt, language="text")


def ajouter_historique(
    question: str,
    reponse_markdown: str,
    resultats: list[rag_pdf.ResultatRecherche],
) -> None:
    """Ajoute une question/reponse en haut de l'historique de session."""
    entree = {
        "question": question,
        "reponse": reponse_markdown,
        "export": construire_export_markdown(question, reponse_markdown, resultats),
        "sources": [
            {
                "numero": index,
                "libelle": libelle_source(resultat.morceau),
                "score": resultat.score_final,
            }
            for index, resultat in enumerate(resultats, start=1)
        ],
    }

    st.session_state.historique_questions.insert(0, entree)
    st.session_state.historique_questions = st.session_state.historique_questions[:10]


def construire_export_markdown(
    question: str,
    reponse_markdown: str,
    resultats: list[rag_pdf.ResultatRecherche],
) -> str:
    """Construit un export Markdown prêt à réutiliser."""
    lignes = [
        "# Réponse RAG",
        "",
        "## Question",
        "",
        question.strip(),
        "",
        "## Réponse",
        "",
        reponse_markdown.strip(),
    ]

    if resultats:
        lignes.extend(["", "## Sources", ""])
        for index, resultat in enumerate(resultats, start=1):
            morceau = resultat.morceau
            url = url_source(morceau) or ""
            lignes.append(
                f"- [{index}] {libelle_source(morceau)}"
                f" | score final {resultat.score_final:.3f}"
                f" | {url}"
            )

    return "\n".join(lignes).strip() + "\n"


def afficher_export_reponse(export_markdown: str) -> None:
    """Affiche les contrôles d'export de la dernière réponse."""
    with st.expander("Exporter la réponse", expanded=False):
        st.download_button(
            "Télécharger en Markdown",
            data=export_markdown,
            file_name="reponse-rag.md",
            mime="text/markdown",
        )
        st.text_area(
            "Texte prêt à copier",
            value=export_markdown,
            height=260,
            help="Sélectionne le texte puis copie-le dans un mail, une note ou un document.",
        )


def afficher_historique() -> None:
    """Affiche les dernieres questions de la session."""
    historique = st.session_state.get("historique_questions", [])

    if not historique:
        return

    with st.expander("Historique de la session", expanded=False):
        if st.button("Vider l'historique"):
            st.session_state.historique_questions = []
            st.rerun()

        for index, entree in enumerate(historique, start=1):
            st.markdown(f"**{index}. {entree['question']}**")
            st.markdown(entree["reponse"])
            if entree.get("export"):
                st.download_button(
                    "Télécharger cette réponse",
                    data=entree["export"],
                    file_name=f"reponse-rag-{index}.md",
                    mime="text/markdown",
                    key=f"download_historique_{index}",
                )
            sources = entree.get("sources", [])
            if sources:
                st.caption(
                    "Sources : "
                    + ", ".join(
                        f"[{source['numero']}] {source['libelle']} ({source['score']:.3f})"
                        for source in sources[:5]
                    )
                )
            st.divider()


def afficher_statistiques() -> None:
    """Affiche l'etat du dossier PDF et de l'index FAISS."""
    stats = statistiques_index_cached(signature_index())

    st.caption(f"Dossier PDF utilisé : `{rag_pdf.PDF_DIR}`")
    st.caption(f"Dossier index utilisé : `{rag_pdf.INDEX_DIR}`")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("PDF dans le dossier", stats["nombre_pdf_dossier"])
    col2.metric("Sources web", stats["nombre_sources_web"])
    col3.metric("Documents indexes", stats["nombre_documents"])
    col4.metric("Chunks indexes", stats["nombre_morceaux"])

    if not stats["index_existe"]:
        st.warning("Aucun index FAISS disponible. Ajoute des PDF puis lance l'indexation.")


def construire_lignes_corpus() -> list[dict[str, str | int | float]]:
    """
    Construit la vue documentaire du corpus.

    Elle compare deux réalités :
    - les PDF présents dans le dossier ;
    - les documents réellement présents dans l'index FAISS.
    """
    pdfs = {pdf.name: pdf for pdf in rag_pdf.lister_pdfs()}
    lignes_par_document: dict[str, dict[str, str | int | float]] = {}

    for nom, chemin in pdfs.items():
        lignes_par_document[nom] = {
            "Document": nom,
            "Type": "PDF",
            "Statut": "À indexer",
            "Taille Mo": round(chemin.stat().st_size / (1024 * 1024), 2),
            "Chunks": 0,
            "Pages": "",
            "Techniques": 0,
            "Tableaux": 0,
        }

    if rag_pdf.METADATA_FILE.exists():
        try:
            _, morceaux = rag_pdf.charger_index()
        except RuntimeError as erreur:
            st.warning(f"Index FAISS temporairement illisible : {erreur}")
            morceaux = []

        for morceau in morceaux:
            source_type = getattr(morceau, "source_type", "pdf")
            nom = morceau.fichier

            if nom not in lignes_par_document:
                lignes_par_document[nom] = {
                    "Document": nom,
                    "Type": "Web" if source_type == "web" else "PDF",
                    "Statut": "Indexé",
                    "Taille Mo": None,
                    "Chunks": 0,
                    "Pages": "",
                    "Techniques": 0,
                    "Tableaux": 0,
                }

            ligne = lignes_par_document[nom]
            ligne["Statut"] = "Indexé"
            ligne["Chunks"] = int(ligne["Chunks"]) + 1

            if source_type == "pdf" and isinstance(morceau.page, int):
                pages = ligne.setdefault("_pages", set())
                if isinstance(pages, set):
                    pages.add(morceau.page)

            texte = morceau.texte.strip().upper()
            if texte.startswith("EXTRAIT TECHNIQUE"):
                ligne["Techniques"] = int(ligne["Techniques"]) + 1
            if texte.startswith("TABLEAU"):
                ligne["Tableaux"] = int(ligne["Tableaux"]) + 1

    lignes = []
    for ligne in lignes_par_document.values():
        pages = ligne.pop("_pages", set())
        if isinstance(pages, set) and pages:
            ligne["Pages"] = f"{min(pages)}-{max(pages)}" if len(pages) > 1 else str(next(iter(pages)))
        lignes.append(ligne)

    return sorted(lignes, key=lambda item: (str(item["Type"]), str(item["Document"]).lower()))


@st.cache_data(show_spinner=False)
def lignes_corpus_cached(_signature: tuple[float, float]) -> list[dict[str, str | int | float]]:
    """Vue documentaire du corpus, cachée car elle lit les métadonnées indexées."""
    return construire_lignes_corpus()


def afficher_bibliotheque_documentaire() -> None:
    """Affiche une vue propre de ce que le RAG connait."""
    lignes = lignes_corpus_cached(signature_index())

    if not lignes:
        st.info("Aucun document dans le dossier ou dans l'index.")
        return

    st.caption("Bibliothèque documentaire")
    st.dataframe(lignes, use_container_width=True, hide_index=True)

    if rag_pdf.FAISS_INDEX_FILE.exists():
        date_index = rag_pdf.FAISS_INDEX_FILE.stat().st_mtime
        st.caption(
            "Dernière indexation FAISS : "
            + datetime.fromtimestamp(date_index).strftime("%Y-%m-%d %H:%M")
        )


def afficher_pdfs_du_dossier() -> None:
    """
    Affiche les PDF réellement présents dans le dossier indexé.

    C'est volontairement explicite : l'utilisateur doit voir exactement quels
    fichiers seront pris en compte par l'indexation.
    """
    pdfs = rag_pdf.lister_pdfs()

    if not pdfs:
        st.info("Aucun PDF présent dans le dossier utilisé par l'indexation.")
        return

    st.caption("PDF actuellement dans le dossier indexé")
    for pdf in pdfs:
        taille_mo = pdf.stat().st_size / (1024 * 1024)
        st.write(f"- `{pdf.name}` ({taille_mo:.2f} Mo)")


def indexer_depuis_ui() -> None:
    """
    Lance l'indexation et capture les messages console.

    rag_pdf.indexer_pdf() ecrit deja des messages pedagogiques avec print().
    On les capture pour les afficher proprement dans Streamlit.
    """
    sortie = io.StringIO()

    with contextlib.redirect_stdout(sortie):
        rag_pdf.indexer_pdf()

    st.code(sortie.getvalue(), language="text")


def indexer_web_depuis_ui(urls: list[str]) -> None:
    """Indexe des pages web controlees puis reconstruit FAISS."""
    sortie = io.StringIO()

    with contextlib.redirect_stdout(sortie):
        rag_pdf.indexer_pages_web(urls)
        rag_pdf.indexer_pdf()

    st.code(sortie.getvalue(), language="text")


def supprimer_pdf_et_reindexer(nom_fichier: str) -> str:
    """Supprime un PDF local puis reconstruit l'index."""
    chemin = rag_pdf.PDF_DIR / nom_fichier
    sortie = io.StringIO()

    with contextlib.redirect_stdout(sortie):
        if chemin.exists():
            chemin.unlink()

        chemin_statique = STATIC_PDF_DIR / nom_fichier
        if chemin_statique.exists():
            chemin_statique.unlink()

        rag_pdf.indexer_pdf()

    return sortie.getvalue()


def supprimer_source_web_et_reindexer(url: str) -> str:
    """Supprime une URL du corpus web local puis reconstruit l'index."""
    sortie = io.StringIO()

    with contextlib.redirect_stdout(sortie):
        morceaux = [
            morceau
            for morceau in rag_pdf.charger_web_chunks()
            if getattr(morceau, "url", "") != url
        ]
        rag_pdf.sauvegarder_web_chunks(morceaux)
        print(f"Source web supprimée : {url}")
        rag_pdf.indexer_pdf()

    return sortie.getvalue()


def construire_filtre(
    documents_selectionnes: list[str],
    page_min: int,
    page_max: int,
    source_types: list[str],
):
    """Transforme les controles UI en FiltreMetadata."""
    fichiers = set(documents_selectionnes) if documents_selectionnes else None
    return rag_pdf.FiltreMetadata(
        fichiers=fichiers,
        page_min=page_min or None,
        page_max=page_max or None,
        source_types=set(source_types) if source_types else None,
    )


def afficher_retrieval_settings() -> None:
    """Réglages retrieval déplacés dans Settings pour libérer la page."""
    with st.expander("Retrieval", expanded=False):
        if cle_openai_presente():
            st.success("Clé OpenAI détectée")
        else:
            st.error("OPENAI_API_KEY absente")
            st.info(
                "Dans le terminal, lance :\n\n"
                'export OPENAI_API_KEY="$(security find-generic-password -a "$USER" -s "OPENAI_API_KEY" -w)"'
            )

        st.slider(
            "Chunks gardés après reranking",
            min_value=1,
            max_value=15,
            key="top_k",
        )

        st.session_state.candidate_k = max(st.session_state.candidate_k, st.session_state.top_k)
        st.slider(
            "Candidats FAISS avant reranking",
            min_value=st.session_state.top_k,
            max_value=200,
            key="candidate_k",
        )

        st.slider(
            "Seuil minimal de score",
            min_value=0.0,
            max_value=0.6,
            step=0.01,
            key="min_score",
        )

        st.slider(
            "Budget contexte approx.",
            min_value=1000,
            max_value=12000,
            step=500,
            key="context_tokens",
        )

        st.checkbox(
            "Query rewriting LLM",
            key="query_rewrite_llm",
            help=(
                "Mode qualité : reformule la question pour la recherche uniquement "
                "(fautes, synonymes FR/EN, acronymes). Ajoute un appel OpenAI, donc "
                "peut ralentir la première réponse."
            ),
        )

        stats_cache = rag_pdf.statistiques_cache_embeddings()
        taille_cache_mo = stats_cache["taille_octets"] / (1024 * 1024)
        st.caption(
            "Cache embeddings : "
            f"{stats_cache['nombre_embeddings']} vecteurs, {taille_cache_mo:.1f} Mo"
        )
        stats_rewrite = rag_pdf.statistiques_cache_query_rewrite()
        taille_rewrite_ko = stats_rewrite["taille_octets"] / 1024
        st.caption(
            "Cache query rewriting : "
            f"{stats_rewrite['nombre_rewrites']} reformulations, {taille_rewrite_ko:.1f} Ko"
        )
        stats_rerank = rag_pdf.statistiques_cache_llm_rerank()
        taille_rerank_ko = stats_rerank["taille_octets"] / 1024
        st.caption(
            "Cache reranker LLM : "
            f"{stats_rerank['nombre_reranks']} classements, {taille_rerank_ko:.1f} Ko"
        )
        st.caption(
            "Le RAG peut élargir automatiquement top_k/candidate_k selon la question."
        )

        if st.button("Réinitialiser les réglages retrieval"):
            st.session_state.top_k = rag_pdf.DEFAULT_TOP_K
            st.session_state.candidate_k = rag_pdf.DEFAULT_CANDIDATE_K
            st.session_state.min_score = rag_pdf.DEFAULT_MIN_SCORE
            st.session_state.context_tokens = rag_pdf.DEFAULT_CONTEXT_TOKEN_BUDGET
            st.session_state.query_rewrite_llm = rag_pdf.DEFAULT_QUERY_REWRITE_LLM
            st.session_state.llm_reranker = rag_pdf.DEFAULT_LLM_RERANKER
            st.session_state.cross_encoder_reranker = rag_pdf.DEFAULT_CROSS_ENCODER_RERANKER
            st.session_state.comparer_reranker = False
            st.rerun()


def afficher_settings_panel() -> None:
    """Affiche les reglages d'administration du corpus."""
    afficher_retrieval_settings()

    with st.expander("Documents", expanded=False):
        st.write(
            "Dépose ici tes PDF. Pour 32 fichiers d'environ quelques Mo au total, "
            "FAISS local reste largement suffisant pour un POC."
        )

        fichiers = st.file_uploader(
            "Ajouter des PDF",
            type=["pdf"],
            accept_multiple_files=True,
        )

        if fichiers and st.button("Ajouter et réindexer les PDF", type="primary"):
            if not cle_openai_presente():
                st.error("Impossible de réindexer sans OPENAI_API_KEY.")
            else:
                chemins = sauver_pdfs(fichiers)
                st.success(f"{len(chemins)} PDF sauvegardé(s). Réindexation automatique...")
                with st.spinner("Reconstruction de l'index FAISS..."):
                    indexer_depuis_ui()
                st.success("Index mis à jour.")
                st.rerun()

        afficher_statistiques()
        afficher_bibliotheque_documentaire()

        if st.checkbox("Afficher la liste simple des PDF", value=False):
            afficher_pdfs_du_dossier()

        pdfs_supprimables = [pdf.name for pdf in rag_pdf.lister_pdfs()]
        if pdfs_supprimables:
            st.divider()
            pdf_a_supprimer = st.selectbox(
                "Supprimer un PDF puis réindexer",
                options=[""] + pdfs_supprimables,
                format_func=lambda valeur: "Choisir un PDF..." if not valeur else valeur,
            )
            if pdf_a_supprimer and st.button("Supprimer ce PDF et réindexer"):
                if not cle_openai_presente():
                    st.error("Impossible de réindexer sans OPENAI_API_KEY.")
                else:
                    with st.spinner("Suppression du PDF et reconstruction de l'index..."):
                        sortie = supprimer_pdf_et_reindexer(pdf_a_supprimer)
                    st.code(sortie, language="text")
                    st.success("PDF supprimé et index mis à jour.")
                    st.rerun()

        if st.button("Forcer une réindexation complète"):
            if not cle_openai_presente():
                st.error("Impossible d'indexer sans OPENAI_API_KEY.")
            else:
                with st.spinner("Indexation en cours..."):
                    indexer_depuis_ui()
                st.success("Indexation terminée.")
                afficher_statistiques()

    with st.expander("Web contrôlé", expanded=False):
        st.write(
            "Cette section n'ouvre pas le modèle à tout Internet. "
            "Elle télécharge uniquement les URLs appartenant aux domaines autorisés, "
            "puis les indexe comme des documents sourcés."
        )

        st.caption("Domaines autorisés : " + ", ".join(sorted(rag_pdf.domaines_web_autorises())))
        st.caption(f"Dossier web local : `{rag_pdf.WEB_DIR}`")

        urls_texte = st.text_area(
            "URLs à indexer, une par ligne",
            placeholder="https://www.purestorage.com/...\nhttps://docs.purestorage.com/...",
            height=140,
        )

        if st.button("Ajouter ces pages web et réindexer", type="primary"):
            urls = [ligne.strip() for ligne in urls_texte.splitlines() if ligne.strip()]

            if not urls:
                st.warning("Ajoute au moins une URL.")
            elif not cle_openai_presente():
                st.error("Impossible d'indexer sans OPENAI_API_KEY.")
            else:
                with st.spinner("Téléchargement contrôlé, extraction, embeddings, FAISS..."):
                    try:
                        indexer_web_depuis_ui(urls)
                    except Exception as erreur:
                        st.error(str(erreur))
                    else:
                        st.success("Sources web indexées.")
                        afficher_statistiques()

        sources_web = rag_pdf.lister_sources_web()

        if sources_web:
            st.caption("Sources web déjà stockées")
            for url in sources_web:
                st.write(f"- {url}")

            url_a_supprimer = st.selectbox(
                "Supprimer une source web puis réindexer",
                options=[""] + sources_web,
                format_func=lambda valeur: "Choisir une URL..." if not valeur else valeur,
            )
            if url_a_supprimer and st.button("Supprimer cette source web et réindexer"):
                if not cle_openai_presente():
                    st.error("Impossible de réindexer sans OPENAI_API_KEY.")
                else:
                    with st.spinner("Suppression de la source web et reconstruction de l'index..."):
                        sortie = supprimer_source_web_et_reindexer(url_a_supprimer)
                    st.code(sortie, language="text")
                    st.success("Source web supprimée et index mis à jour.")
                    st.rerun()
        else:
            st.info("Aucune source web stockée pour l'instant.")

    with st.expander("Debug", expanded=False):
        st.checkbox(
            "Afficher le debug après chaque réponse",
            key="mode_debug",
            help="Montre la requête enrichie, les chunks récupérés et les scores.",
        )
        st.markdown(
            """
            **Pipeline**

            1. Les PDF sont lus page par page avec PyMuPDF.
            2. Les tableaux détectés sont conservés comme chunks Markdown séparés.
            3. Le texte est découpé par phrases pour éviter de casser les idées.
            4. La question peut être reformulée par LLM pour améliorer la recherche.
            5. OpenAI transforme chaque chunk en embedding, avec cache local.
            6. FAISS stocke les vecteurs et retrouve les chunks proches de la question.
            7. Un reranking hybride combine score sémantique FAISS et score lexical BM25.
            8. Le `top_k` est adapté selon le type de question.
            9. Le prompt garde uniquement les chunks sous le budget de contexte.

            **Conseil de debug**

            Si la réponse est trop souvent “Je ne sais pas”, baisse le seuil minimal,
            par exemple `0.00`, puis regarde les sources.
            """
        )

        st.code(
            "streamlit run app_streamlit.py\n"
            "python rag_pdf.py demander \"Ta question\" --min_score 0 --sources",
            language="bash",
        )

initialiser_etat_session()

col_title, col_settings = st.columns([0.88, 0.12], vertical_alignment="top")

with col_title:
    st.markdown(
        """
        <div class="everpure-hero">
            <div class="everpure-kicker">Enterprise Data Cloud RAG</div>
            <h1>Everpure</h1>
            <p>Assistant documentaire avec sources cliquables, retrieval hybride FAISS + BM25, et contrôle du corpus.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_settings:
    if st.button("⚙", help="Settings", key="settings_button"):
        st.session_state.afficher_settings = not st.session_state.get("afficher_settings", False)

if st.session_state.get("afficher_settings", False):
    with st.container(border=True):
        col_settings_title, col_settings_close = st.columns([0.9, 0.1], vertical_alignment="center")
        with col_settings_title:
            st.subheader("Settings")
        with col_settings_close:
            if st.button("Fermer"):
                st.session_state.afficher_settings = False
                st.rerun()
        afficher_settings_panel()

st.subheader("Interroger les documents")

top_k = st.session_state.top_k
candidate_k = st.session_state.candidate_k
min_score = st.session_state.min_score
context_tokens = st.session_state.context_tokens

question = st.text_area(
    "Question",
    placeholder="Exemple : Compare FlashArray et FlashBlade.",
    height=100,
)

modeles_generation = list(rag_pdf.DEFAULT_CHAT_MODELS)

if st.session_state.modele_generation not in modeles_generation:
    # Ancien choix invalide ou modèle retiré côté API.
    # On revient au modèle stable par défaut au lieu de garder une valeur qui
    # provoquerait une erreur OpenAI au moment de générer la réponse.
    st.session_state.modele_generation = rag_pdf.DEFAULT_CHAT_MODEL

modele_generation = st.selectbox(
    "Modèle de réponse",
    options=modeles_generation,
    key="modele_generation",
    help=(
        "Ce choix change uniquement le modèle qui rédige la réponse finale. "
        "Le retrieval reste identique : mêmes embeddings, même FAISS, mêmes sources."
    ),
)

afficher_historique()

with st.expander("Options avancées : sources, documents et pages", expanded=False):
    source_types = st.multiselect(
        "Sources à utiliser",
        options=["pdf", "web"],
        key="source_types",
        format_func=lambda valeur: "PDF locaux" if valeur == "pdf" else "Web contrôlé",
        help="Décoche PDF pour forcer une réponse uniquement depuis les pages web indexées.",
    )

    col_reranker, col_cross_encoder, col_compare = st.columns(3)
    with col_reranker:
        st.checkbox(
            "Reranker LLM lent",
            key="llm_reranker",
            help=(
                "Mode expérimental : après FAISS + BM25, demande au LLM de choisir "
                "les chunks les plus utiles. Peut prendre plus d'une minute selon "
                "le nombre de candidats."
            ),
        )
    with col_cross_encoder:
        st.checkbox(
            "Reranker BGE local",
            key="cross_encoder_reranker",
            help=(
                f"Utilise un CrossEncoder local {rag_pdf.DEFAULT_CROSS_ENCODER_MODEL} "
                "sur les meilleurs chunks. S'il n'est pas réellement prêt, l'app revient "
                "automatiquement à FAISS + BM25."
            ),
        )
    with col_compare:
        st.checkbox(
            "Comparer avec / sans",
            key="comparer_reranker",
            help="Compare le ranking de base FAISS + BM25 avec les rerankers avancés sélectionnés.",
        )

    documents_indexes = documents_indexes_cached(signature_index())
    documents_selectionnes = st.multiselect(
        "Filtrer par document",
        options=documents_indexes,
        key="documents_selectionnes",
        help="Laisse vide pour chercher dans tous les documents indexés.",
    )

    col_page_min, col_page_max = st.columns(2)
    page_min = col_page_min.number_input(
        "Page min",
        min_value=0,
        key="page_min",
        step=1,
        help="0 signifie : pas de limite basse.",
    )
    page_max = col_page_max.number_input(
        "Page max",
        min_value=0,
        key="page_max",
        step=1,
        help="0 signifie : pas de limite haute.",
    )

    if st.button("Réinitialiser les filtres"):
        st.session_state.source_types = ["pdf", "web"]
        st.session_state.documents_selectionnes = []
        st.session_state.page_min = 0
        st.session_state.page_max = 0
        st.rerun()

source_types = st.session_state.source_types
documents_selectionnes = st.session_state.documents_selectionnes
page_min = st.session_state.page_min
page_max = st.session_state.page_max
query_rewrite_llm = st.session_state.query_rewrite_llm
llm_reranker = st.session_state.llm_reranker
cross_encoder_reranker = st.session_state.cross_encoder_reranker
comparer_reranker = st.session_state.comparer_reranker
modele_generation = st.session_state.modele_generation

source_labels = {
    "pdf": "PDF locaux",
    "web": "Web contrôlé",
}
sources_actives = ", ".join(source_labels[source] for source in source_types) or "aucune"
st.caption(f"Sources actives : {sources_actives}")
st.caption(
    f"Réglages retrieval de base : top_k={top_k}, candidate_k={candidate_k}, "
    f"min_score={min_score:.2f}, contexte={context_tokens} tokens, "
    f"modèle réponse={modele_generation}, "
    f"query rewrite LLM={'on' if query_rewrite_llm else 'off'}, "
    f"reranker LLM={'on' if llm_reranker else 'off'}, "
    f"BGE local={'on' if cross_encoder_reranker else 'off'}"
)

if cross_encoder_reranker:
    st.caption(
        "BGE demandé : l'interface affichera après la réponse si BGE a réellement "
        "reranké ou si le pipeline est revenu à FAISS + BM25. Premier appel BGE : "
        "chargement local du modèle, souvent 30 à 60 secondes ; appels suivants : "
        "beaucoup plus rapides."
    )

if cross_encoder_reranker and not rag_pdf.cross_encoder_disponible():
    st.warning(
        "Le reranker BGE local est coché, mais il n'est pas activé dans cet environnement. "
        "Le RAG utilise donc FAISS + BM25 + heuristiques pour préserver la stabilité. "
        "Pour BGE stable, le projet utilise le worker séparé `.venv311/bin/python scripts/bge_worker.py`."
    )

if "pdf" not in source_types:
    st.warning(
        "PDF locaux est désactivé. Les questions sur les datasheets PDF, comme FlashArray XL, "
        "ne pourront pas retrouver les valeurs présentes dans les PDF."
    )

if st.button("Demander", type="primary"):
    if not cle_openai_presente():
        st.error("OPENAI_API_KEY absente.")
    elif not question.strip():
        st.warning("Écris une question.")
    elif not source_types:
        st.warning("Sélectionne au moins une source : PDF locaux ou Web contrôlé.")
    else:
        question_technique = rag_pdf.question_demande_valeurs_techniques(question)
        parametres_adaptatifs = rag_pdf.calculer_parametres_recherche_adaptatifs(
            question=question,
            top_k=top_k,
            candidate_k=candidate_k,
            min_score=min_score,
        )
        top_k_effectif = parametres_adaptatifs.top_k
        candidate_k_effectif = parametres_adaptatifs.candidate_k
        min_score_effectif = parametres_adaptatifs.min_score
        raison_adaptation = parametres_adaptatifs.raison

        if question_technique:
            filtre = construire_filtre(
                documents_selectionnes=[],
                page_min=0,
                page_max=0,
                source_types=["pdf", "web"],
            )
        else:
            filtre = construire_filtre(documents_selectionnes, page_min, page_max, source_types)

        fallback_utilise = False
        resultats_sans_reranker = []
        resultats_avec_reranker = []
        temps_debut_total = time.perf_counter()
        temps_retrieval_s = 0.0
        temps_generation_s = 0.0

        with st.spinner("Recherche des passages, reranking, puis génération..."):
            temps_debut_retrieval = time.perf_counter()
            resultats = rag_pdf.rechercher(
                question=question,
                top_k=top_k_effectif,
                candidate_k=candidate_k_effectif,
                min_score=min_score_effectif,
                filtre=filtre,
                adaptatif=False,
                utiliser_query_rewrite_llm=query_rewrite_llm,
                utiliser_reranker_llm=llm_reranker,
                utiliser_cross_encoder_reranker=cross_encoder_reranker,
            )
            temps_retrieval_s += time.perf_counter() - temps_debut_retrieval

            if comparer_reranker:
                temps_debut_comparaison = time.perf_counter()
                resultats_avec_reranker = resultats
                resultats_sans_reranker = rag_pdf.rechercher(
                    question=question,
                    top_k=top_k_effectif,
                    candidate_k=candidate_k_effectif,
                    min_score=min_score_effectif,
                    filtre=filtre,
                    adaptatif=False,
                    utiliser_query_rewrite_llm=query_rewrite_llm,
                    utiliser_reranker_llm=False,
                    utiliser_cross_encoder_reranker=False,
                )
                temps_retrieval_s += time.perf_counter() - temps_debut_comparaison

            if not resultats and question_technique:
                fallback_utilise = True
                filtre_fallback = construire_filtre(
                    documents_selectionnes=[],
                    page_min=0,
                    page_max=0,
                    source_types=["pdf", "web"],
                )
                temps_debut_fallback = time.perf_counter()
                resultats = rag_pdf.rechercher(
                    question=question,
                    top_k=max(top_k, 8),
                    candidate_k=max(candidate_k, 120),
                    min_score=0,
                    filtre=filtre_fallback,
                    adaptatif=True,
                    utiliser_query_rewrite_llm=query_rewrite_llm,
                    utiliser_reranker_llm=llm_reranker,
                    utiliser_cross_encoder_reranker=cross_encoder_reranker,
                )
                temps_retrieval_s += time.perf_counter() - temps_debut_fallback
                resultats_sans_reranker = []
                resultats_avec_reranker = []

            prompt = rag_pdf.construire_prompt(
                question=question,
                resultats=resultats,
                budget_tokens=context_tokens,
            )
            temps_debut_generation = time.perf_counter()
            try:
                reponse = rag_pdf.appeler_modele(prompt, modele=modele_generation)
            except Exception as erreur:
                st.error(
                    "Le modèle de réponse sélectionné n'a pas pu être appelé. "
                    "Essaie un autre modèle dans la liste."
                )
                st.exception(erreur)
                st.stop()
            temps_generation_s = time.perf_counter() - temps_debut_generation

        trace_rag = {
            "modele_generation": modele_generation,
            "top_k_effectif": top_k_effectif,
            "candidate_k_effectif": candidate_k_effectif,
            "min_score_effectif": min_score_effectif,
            "context_tokens": context_tokens,
            "query_rewrite_llm": query_rewrite_llm,
            "llm_reranker": llm_reranker,
            "cross_encoder_reranker": cross_encoder_reranker,
            "bge_actif": rag_pdf.dernier_reranker_bge_actif(),
            "raison_adaptation": raison_adaptation,
            "fallback_utilise": fallback_utilise,
            "temps_retrieval_s": temps_retrieval_s,
            "temps_generation_s": temps_generation_s,
            "temps_total_s": time.perf_counter() - temps_debut_total,
            "requetes_recherche": rag_pdf.construire_requetes_recherche(
                question,
                utiliser_query_rewrite_llm=query_rewrite_llm,
            ),
        }

        st.markdown("### Réponse")
        st.caption(f"Réponse générée avec : `{modele_generation}`")
        if question_technique:
            st.info(
                "Question technique détectée : recherche élargie automatiquement "
                "sur PDF + Web avec min_score=0 pour éviter de masquer les tableaux de specs."
            )
        if raison_adaptation:
            st.caption(f"Adaptation retrieval : {raison_adaptation}")
        if fallback_utilise:
            st.info(
                "Aucun résultat avec les filtres/réglages courants. "
                "Fallback automatique utilisé : PDF + Web, min_score=0, candidate_k=120."
            )
        if comparer_reranker and resultats_sans_reranker and resultats_avec_reranker:
            afficher_comparaison_reranker(resultats_sans_reranker, resultats_avec_reranker)
        if cross_encoder_reranker:
            if rag_pdf.dernier_reranker_bge_actif():
                st.success("BGE actif : les chunks ont été rerankés par le modèle local.")
            else:
                st.warning(
                    "BGE n'a pas réellement reranké cette réponse. "
                    "Fallback utilisé : FAISS + BM25 + heuristiques."
                )
        reponse_markdown = afficher_reponse_avec_sources(reponse, resultats)
        afficher_trace_rag(trace_rag, resultats, prompt)
        export_markdown = construire_export_markdown(question, reponse_markdown, resultats)
        st.session_state.dernier_export_markdown = export_markdown
        ajouter_historique(question, reponse_markdown, resultats)
        afficher_export_reponse(export_markdown)

        if not resultats:
            st.warning(f"Aucun chunk au-dessus du seuil {min_score:.2f}. Essaie 0.00.")

        if st.session_state.mode_debug:
            afficher_debug_recherche(
                question=question,
                resultats=resultats,
                top_k_effectif=top_k_effectif,
                candidate_k_effectif=candidate_k_effectif,
                min_score_effectif=min_score_effectif,
                context_tokens=context_tokens,
                raison_adaptation=raison_adaptation,
                query_rewrite_llm=query_rewrite_llm,
                llm_reranker=llm_reranker,
                cross_encoder_reranker=cross_encoder_reranker,
            )
