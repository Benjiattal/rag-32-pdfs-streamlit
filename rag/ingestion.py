"""
Ingestion des sources.

Responsabilite :
lire les PDF, extraire les tableaux, recuperer les pages web autorisees,
construire l'index FAISS local et le recharger.

Principe RAG :
l'ingestion est la partie qui transforme des documents bruts en corpus
interrogeable. C'est une etape separee de la generation : on peut reconstruire
l'index sans poser de question au modele final.
"""

from __future__ import annotations

from datetime import date
import io
import os
import pickle
import re
import time
from pathlib import Path
from urllib.parse import urlparse

from rag.chunking import decouper_texte, decouper_texte_par_sections
from rag.config import (
    PDF_DIR,
    INDEX_DIR,
    WEB_DIR,
    FAISS_INDEX_FILE,
    METADATA_FILE,
    WEB_CHUNKS_FILE,
    DEFAULT_ALLOWED_WEB_DOMAINS,
)
from rag.env import charger_env_local
from rag.models import Morceau


def lister_pdfs() -> list[Path]:
    """Retourne la liste des PDF disponibles dans data/pdfs."""
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(PDF_DIR.glob("*.pdf"))


def domaines_web_autorises() -> set[str]:
    """
    Retourne les domaines autorises pour le RAG web.

    Principe qualite :
    on n'ouvre pas le modele a tout Internet. On limite volontairement les
    sources a une whitelist de domaines fiables.
    """
    domaines = os.getenv("RAG_ALLOWED_WEB_DOMAINS", DEFAULT_ALLOWED_WEB_DOMAINS)
    return {domaine.strip().lower() for domaine in domaines.split(",") if domaine.strip()}


def domaine_est_autorise(url: str) -> bool:
    """Verifie si l'URL appartient a un domaine autorise."""
    domaine = urlparse(url).netloc.lower()

    if domaine.startswith("www."):
        domaine = domaine[4:]

    for domaine_autorise in domaines_web_autorises():
        if domaine == domaine_autorise or domaine.endswith("." + domaine_autorise):
            return True

    return False


def lister_sources_web() -> list[str]:
    """Retourne les URLs deja stockees dans le corpus web local."""
    morceaux = charger_web_chunks()
    return sorted({morceau.url for morceau in morceaux if getattr(morceau, "url", "")})


def lister_documents_indexes() -> list[str]:
    """
    Retourne les noms des documents presents dans l'index.

    Utile pour une UI : on peut proposer un filtre par document sans relire les
    PDF ni recalculer les embeddings.
    """
    if not METADATA_FILE.exists():
        return []

    try:
        _, morceaux = charger_index()
    except RuntimeError:
        return []

    return sorted({morceau.fichier for morceau in morceaux})


def statistiques_index() -> dict[str, int | bool]:
    """
    Donne quelques informations simples sur l'index courant.

    Principe : une UI ou un utilisateur CLI doit pouvoir savoir rapidement si
    l'index existe, combien de documents il couvre, et combien de chunks il
    contient.
    """
    stats: dict[str, int | bool] = {
        "index_existe": FAISS_INDEX_FILE.exists() and METADATA_FILE.exists(),
        "nombre_documents": 0,
        "nombre_morceaux": 0,
        "nombre_pdf_dossier": len(lister_pdfs()),
        "nombre_sources_web": len(lister_sources_web()),
    }

    if not stats["index_existe"]:
        return stats

    try:
        _, morceaux = charger_index()
    except RuntimeError:
        stats["index_existe"] = False
        return stats

    stats["nombre_morceaux"] = len(morceaux)
    stats["nombre_documents"] = len({morceau.fichier for morceau in morceaux})

    return stats


class UnpicklerCompatibleRag(pickle.Unpickler):
    """
    Charge les anciens fichiers pickle crees avec `python rag_pdf.py`.

    Pourquoi c'est necessaire ?
    Quand un fichier Python est lance directement, ses classes vivent dans le
    module special `__main__`. Si ensuite Streamlit importe ce meme fichier sous
    le nom `rag_pdf`, pickle cherche `__main__.Morceau` et ne le trouve pas.
    """

    def find_class(self, module: str, name: str):
        if module == "__main__" and name == "Morceau":
            return Morceau

        return super().find_class(module, name)


def charger_morceaux_pickle(chemin: Path) -> list[Morceau]:
    """
    Charge les metadonnees de chunks avec compatibilite Streamlit/CLI.

    A moyen terme, JSON serait plus portable que pickle. Pour ce POC, on garde
    pickle mais on evite que l'UI casse sur les index deja crees.
    """
    donnees = chemin.read_bytes()

    try:
        return pickle.loads(donnees)
    except AttributeError:
        return UnpicklerCompatibleRag(io.BytesIO(donnees)).load()


def charger_web_chunks() -> list[Morceau]:
    """Charge les chunks web deja recuperes localement."""
    if not WEB_CHUNKS_FILE.exists():
        return []

    return charger_morceaux_pickle(WEB_CHUNKS_FILE)


def sauvegarder_web_chunks(morceaux: list[Morceau]) -> None:
    """Sauvegarde les chunks web dans data/web."""
    WEB_DIR.mkdir(parents=True, exist_ok=True)

    with WEB_CHUNKS_FILE.open("wb") as fichier:
        pickle.dump(morceaux, fichier)


def lire_pdf(fichier_pdf: Path) -> list[Morceau]:
    """
    Lit un PDF et retourne une liste de morceaux de texte.

    Ici on utilise PyMuPDF, importe sous le nom "fitz".
    Il extrait generalement mieux le texte que beaucoup de lecteurs PDF simples.
    """
    try:
        import fitz  # PyMuPDF
    except ModuleNotFoundError as erreur:
        raise ModuleNotFoundError(
            "La bibliotheque PyMuPDF manque. Lance d'abord : "
            "pip install -r requirements.txt"
        ) from erreur

    morceaux: list[Morceau] = []

    with fitz.open(fichier_pdf) as document:
        for numero_page, page in enumerate(document, start=1):
            numero_morceau_page = 1

            for tableau in extraire_tableaux_page(page):
                morceaux.append(
                    Morceau(
                        texte=tableau,
                        fichier=fichier_pdf.name,
                        page=numero_page,
                        numero=numero_morceau_page,
                        titre="Tableau",
                    )
                )
                numero_morceau_page += 1

            texte = page.get_text("text").strip()

            if not texte:
                continue

            for extrait_technique in extraire_extraits_techniques(texte, numero_page):
                morceaux.append(
                    Morceau(
                        texte=extrait_technique,
                        fichier=fichier_pdf.name,
                        page=numero_page,
                        numero=numero_morceau_page,
                        titre="Extrait technique",
                    )
                )
                numero_morceau_page += 1

            morceaux_page = decouper_texte_par_sections(texte)

            for morceau, titre_section in morceaux_page:
                morceaux.append(
                    Morceau(
                        texte=morceau,
                        fichier=fichier_pdf.name,
                        page=numero_page,
                        numero=numero_morceau_page,
                        titre=titre_section,
                    )
                )
                numero_morceau_page += 1

    return morceaux


def extraire_tableaux_page(page) -> list[str]:
    """
    Extrait les tableaux detectes par PyMuPDF sous forme Markdown.

    Les datasheets contiennent souvent les valeurs importantes dans des tableaux
    : watts, capacites, SLA, dimensions. Un chunk tableau separe donne au
    retrieval et au modele une structure beaucoup plus claire.
    """
    if not hasattr(page, "find_tables"):
        return []

    try:
        tables = page.find_tables()
    except Exception:
        return []

    morceaux_tableaux: list[str] = []

    for index_tableau, table in enumerate(getattr(tables, "tables", []), start=1):
        lignes = table.extract()
        markdown = convertir_tableau_en_markdown(lignes)

        if not markdown:
            continue

        morceaux_tableaux.append(
            f"TABLEAU extrait de la page {page.number + 1}, tableau {index_tableau}\n\n"
            f"{markdown}"
        )

    return morceaux_tableaux


def convertir_tableau_en_markdown(lignes: list[list[object]]) -> str:
    """Convertit une table PyMuPDF en Markdown simple."""
    lignes_nettoyees: list[list[str]] = []

    for ligne in lignes:
        cellules = [" ".join(str(cellule or "").split()) for cellule in ligne]

        if any(cellules):
            lignes_nettoyees.append(cellules)

    if len(lignes_nettoyees) < 2:
        return ""

    largeur = max(len(ligne) for ligne in lignes_nettoyees)
    lignes_normalisees = [ligne + [""] * (largeur - len(ligne)) for ligne in lignes_nettoyees]
    entete = lignes_normalisees[0]

    if not any(entete):
        entete = [f"Colonne {index}" for index in range(1, largeur + 1)]

    lignes_markdown = [
        "| " + " | ".join(entete) + " |",
        "| " + " | ".join(["---"] * largeur) + " |",
    ]

    for ligne in lignes_normalisees[1:]:
        lignes_markdown.append("| " + " | ".join(ligne) + " |")

    return "\n".join(lignes_markdown)


def extraire_extraits_techniques(texte: str, numero_page: int) -> list[str]:
    """
    Cree des chunks courts autour des lignes qui portent des valeurs techniques.

    Certains PDF ne sont pas detectes comme tableaux par PyMuPDF, mais leur texte
    garde quand meme une ligne par valeur : watts, PB, GB/s, latency, SLA, etc.
    """
    lignes = [" ".join(ligne.split()) for ligne in texte.splitlines()]
    lignes = [ligne for ligne in lignes if ligne]

    motif_unite = re.compile(
        r"(\d[\d,.\s\u2013-]*(?:watts?|w|pb|pib|tb|tib|gb/s|gb|iops|ms|\u00b5s|us|sla|u)\b)",
        flags=re.IGNORECASE,
    )
    indices_utiles = [
        index for index, ligne in enumerate(lignes) if motif_unite.search(ligne)
    ]

    if len(indices_utiles) < 2:
        return []

    fenetres: list[tuple[int, int]] = []
    for index in indices_utiles:
        debut = max(0, index - 4)
        fin = min(len(lignes), index + 3)

        if fenetres and debut <= fenetres[-1][1]:
            fenetres[-1] = (fenetres[-1][0], max(fenetres[-1][1], fin))
        else:
            fenetres.append((debut, fin))

    extraits = []
    for numero, (debut, fin) in enumerate(fenetres[:4], start=1):
        bloc = "\n".join(lignes[debut:fin])

        if len(bloc) < 80:
            continue

        extraits.append(
            f"EXTRAIT TECHNIQUE page {numero_page}, bloc {numero}\n\n{bloc}"
        )

    return extraits


def lire_page_web(url: str) -> list[Morceau]:
    """
    Telecharge une page web autorisee et la transforme en chunks.

    Controle qualite :
    - l'URL doit appartenir a un domaine autorise ;
    - le HTML est nettoye pour retirer scripts, styles et navigation parasite ;
    - chaque chunk garde l'URL et la date de consultation.
    """
    if not domaine_est_autorise(url):
        domaines = ", ".join(sorted(domaines_web_autorises()))
        raise ValueError(f"Domaine non autorise pour {url}. Domaines autorises : {domaines}")

    try:
        import requests
        from bs4 import BeautifulSoup
    except ModuleNotFoundError as erreur:
        raise ModuleNotFoundError(
            "Les bibliotheques web manquent. Lance : pip install -r requirements.txt"
        ) from erreur

    reponse = requests.get(
        url,
        timeout=20,
        headers={"User-Agent": "MiniRAG/1.0"},
    )
    reponse.raise_for_status()

    soup = BeautifulSoup(reponse.text, "html.parser")

    for element in soup(["script", "style", "noscript", "svg", "header", "footer", "nav"]):
        element.decompose()

    titre = soup.title.get_text(" ", strip=True) if soup.title else url
    texte = soup.get_text(" ", strip=True)
    texte = " ".join(texte.split())

    if not texte:
        return []

    morceaux = []
    domaine = urlparse(url).netloc
    date_du_jour = date.today().isoformat()

    for numero_morceau, morceau in enumerate(decouper_texte(texte), start=1):
        morceaux.append(
            Morceau(
                texte=morceau,
                fichier=titre or domaine,
                page=0,
                numero=numero_morceau,
                source_type="web",
                url=url,
                titre=titre,
                date_consultation=date_du_jour,
            )
        )

    return morceaux


def indexer_pages_web(urls: list[str]) -> list[Morceau]:
    """
    Recupere des pages web controlees et les stocke localement.

    Les nouveaux chunks remplacent les anciens chunks portant la meme URL.
    Ensuite, il faut reconstruire l'index FAISS global avec indexer_pdf().
    """
    WEB_DIR.mkdir(parents=True, exist_ok=True)
    chunks_existants = charger_web_chunks()
    urls_normalisees = {url.strip() for url in urls if url.strip()}
    nouveaux_chunks: list[Morceau] = []

    for url in sorted(urls_normalisees):
        print(f"Lecture web : {url}")
        morceaux_url = lire_page_web(url)
        nouveaux_chunks.extend(morceaux_url)
        print(f"- {len(morceaux_url)} morceaux web")

    chunks_conserves = [
        morceau for morceau in chunks_existants if getattr(morceau, "url", "") not in urls_normalisees
    ]
    tous_les_chunks = chunks_conserves + nouveaux_chunks
    sauvegarder_web_chunks(tous_les_chunks)

    print(f"Sources web sauvegardees : {WEB_CHUNKS_FILE}")
    print(f"Nombre total de morceaux web : {len(tous_les_chunks)}")

    return nouveaux_chunks


def indexer_pdf() -> None:
    """
    Cree l'index vectoriel FAISS a partir des PDF et sources web locales.

    Cette etape est a lancer quand :
    - tu ajoutes des PDF ;
    - tu supprimes des PDF ;
    - tu modifies les documents source ;
    - tu ajoutes ou retires des pages web controlees.
    """
    charger_env_local()
    from rag.embeddings import creer_embeddings, statistiques_cache_embeddings

    PDF_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    fichiers_pdf = sorted(PDF_DIR.glob("*.pdf"))
    morceaux_web = charger_web_chunks()

    if not fichiers_pdf and not morceaux_web:
        print(f"Aucun PDF trouve dans : {PDF_DIR}")
        print("Aucune source web trouvee non plus.")
        print("Ajoute des PDF ou indexe des URLs web controlees, puis relance l'indexation.")
        return

    tous_les_morceaux: list[Morceau] = []

    print("Lecture des PDF...")
    for fichier_pdf in fichiers_pdf:
        morceaux_pdf = lire_pdf(fichier_pdf)
        tous_les_morceaux.extend(morceaux_pdf)
        print(f"- {fichier_pdf.name} : {len(morceaux_pdf)} morceaux")

    if morceaux_web:
        tous_les_morceaux.extend(morceaux_web)
        print(f"Sources web locales : {len(morceaux_web)} morceaux")

    if not tous_les_morceaux:
        print("Aucun texte lisible n'a ete trouve dans les PDF.")
        print("Certains PDF scannes sous forme d'image necessitent de l'OCR.")
        return

    try:
        import faiss
    except ModuleNotFoundError as erreur:
        raise ModuleNotFoundError(
            "La bibliotheque FAISS manque. Lance d'abord : "
            "pip install -r requirements.txt"
        ) from erreur

    textes = [morceau.texte for morceau in tous_les_morceaux]

    stats_cache_avant = statistiques_cache_embeddings()
    print("Creation ou recuperation des embeddings OpenAI...")
    print(
        "Cache embeddings avant : "
        f"{stats_cache_avant['nombre_embeddings']} vecteurs"
    )
    vecteurs = creer_embeddings(textes)
    stats_cache_apres = statistiques_cache_embeddings()
    print(
        "Cache embeddings apres : "
        f"{stats_cache_apres['nombre_embeddings']} vecteurs"
    )

    dimension = vecteurs.shape[1]

    # IndexFlatIP = recherche par produit scalaire.
    # Comme les vecteurs sont normalises, cela revient a chercher par similarite
    # cosinus.
    index = faiss.IndexFlatIP(dimension)
    index.add(vecteurs)

    index_temporaire = FAISS_INDEX_FILE.with_suffix(".faiss.tmp")
    metadata_temporaire = METADATA_FILE.with_suffix(".pkl.tmp")

    faiss.write_index(index, str(index_temporaire))

    with metadata_temporaire.open("wb") as fichier:
        pickle.dump(tous_les_morceaux, fichier)

    os.replace(index_temporaire, FAISS_INDEX_FILE)
    os.replace(metadata_temporaire, METADATA_FILE)

    print(f"Index FAISS cree : {FAISS_INDEX_FILE}")
    print(f"Sources sauvegardees : {METADATA_FILE}")
    print(f"Nombre total de morceaux : {len(tous_les_morceaux)}")


def charger_index():
    """Charge l'index FAISS et les metadonnees."""
    try:
        import faiss
    except ModuleNotFoundError as erreur:
        raise ModuleNotFoundError(
            "La bibliotheque FAISS manque. Lance d'abord : "
            "pip install -r requirements.txt"
        ) from erreur

    if not FAISS_INDEX_FILE.exists() or not METADATA_FILE.exists():
        raise FileNotFoundError(
            "L'index n'existe pas encore. Lance d'abord : python rag_pdf.py indexer"
        )

    derniere_erreur: Exception | None = None

    for tentative in range(1, 4):
        try:
            index = faiss.read_index(str(FAISS_INDEX_FILE))
            morceaux = charger_morceaux_pickle(METADATA_FILE)
            return index, morceaux
        except (RuntimeError, OSError, EOFError, pickle.UnpicklingError) as erreur:
            derniere_erreur = erreur
            if tentative < 3:
                time.sleep(0.4 * tentative)

    raise RuntimeError(
        "Impossible de lire l'index FAISS apres plusieurs tentatives. "
        "Il est peut-etre en cours de reindexation ou doit etre reconstruit "
        "avec : python rag_pdf.py indexer"
    ) from derniere_erreur
