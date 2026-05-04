"""
Construction du contexte et du prompt final.

Responsabilite :
transformer les resultats du retrieval en contexte lisible pour le modele de
reponse, puis injecter les regles generiques et les consignes du profil actif.

Pourquoi ce module existe ?
- Le moteur `rag.engine` doit rester un orchestrateur.
- Les regles de prompt evoluent souvent pendant les demos.
- Les sources et citations sont critiques pour la confiance Sales Engineering.
"""

from __future__ import annotations

from rag.config import DEFAULT_CONTEXT_TOKEN_BUDGET
from rag.everpure import construire_aides_prompt as construire_aides_prompt_everpure
from rag.models import ResultatRecherche
from rag.profiles import profil_domaine_actif


def estimer_tokens(texte: str) -> int:
    """
    Estime grossierement le nombre de tokens.

    Regle pratique : en francais/anglais, 1 token vaut souvent environ
    4 caracteres. Ce n'est pas parfait, mais suffisant pour gerer un budget de
    contexte dans un script pedagogique.
    """
    return max(1, len(texte) // 4)


def construire_contexte_long(
    resultats: list[ResultatRecherche],
    budget_tokens: int,
) -> str:
    """
    Construit un contexte en respectant un budget.

    Principe suivi :
    - on garde les meilleurs resultats rerankes ;
    - on evite de depasser un contexte trop long ;
    - on groupe les sources par document pour aider le multi-doc reasoning.
    """
    blocs_par_document: dict[str, list[str]] = {}
    tokens_utilises = 0

    for numero, resultat in enumerate(resultats, start=1):
        morceau = resultat.morceau
        source_type = getattr(morceau, "source_type", "pdf")
        if source_type == "web":
            localisation = (
                f"url {getattr(morceau, 'url', '')}, "
                f"consulte le {getattr(morceau, 'date_consultation', '')}"
            )
        else:
            localisation = f"page {morceau.page}"

        bloc = (
            f"[Source {numero} | Document : {morceau.fichier} | {localisation}, "
            f"chunk {getattr(morceau, 'numero', '?')}, "
            f"score final {resultat.score_final:.3f}, "
            f"score FAISS {resultat.score_semantique:.3f}, "
            f"score lexical {resultat.score_lexical:.3f}]\n"
            f"{morceau.texte}"
        )
        cout = estimer_tokens(bloc)

        if tokens_utilises + cout > budget_tokens:
            continue

        blocs_par_document.setdefault(morceau.fichier, []).append(bloc)
        tokens_utilises += cout

    morceaux_contexte = []

    for fichier, blocs in blocs_par_document.items():
        morceaux_contexte.append(f"## Document : {fichier}\n" + "\n\n".join(blocs))

    return "\n\n".join(morceaux_contexte)


def construire_prompt(
    question: str,
    resultats: list[ResultatRecherche],
    budget_tokens: int = DEFAULT_CONTEXT_TOKEN_BUDGET,
) -> str:
    """
    Construit le prompt envoye au modele de reponse.
    """
    contexte = construire_contexte_long(resultats, budget_tokens=budget_tokens)

    if not contexte:
        contexte = "Aucun extrait suffisamment pertinent n'a ete retrouve."

    profil = profil_domaine_actif()
    aides_profil = ""

    if profil.name == "everpure":
        aides_profil = construire_aides_prompt_everpure(question, resultats)

    consignes_profil = profil.prompt_extra_text()
    if consignes_profil:
        consignes_profil = "\n" + consignes_profil + "\n"

    prompt = f"""
Tu es un assistant RAG.

Reponds a la question en utilisant UNIQUEMENT le contexte fourni.

Regles :
- Reponds toujours en francais, meme si les sources sont en anglais.
- Si la reponse n'est pas dans le contexte, dis : "Je ne sais pas."
- N'invente pas d'information absente du contexte.
- Ne complete pas avec tes connaissances generales.
- N'utilise pas les documents comme simple inspiration : chaque affirmation importante doit venir du contexte.
- Pour une question demandant une liste de valeurs techniques, reponds sous forme de tableau avec les colonnes pertinentes et cite la source.
- Cite les sources avec des numeros entre crochets, par exemple [1] ou [2], en utilisant les numeros "Source N" du contexte.
- Ne recopie pas le nom complet du fichier dans la reponse si une citation [N] suffit.
- Si le contexte contient des chiffres exacts, recopie-les exactement sans les arrondir.
- Si le contexte contient une plage de valeurs correspondant a la question, ne dis pas que la liste est absente : extrais ces valeurs.
- Pour une question de valeurs techniques, commence directement par le tableau. N'ecris pas "le contexte ne fournit pas" si au moins une valeur est presente.
- Evite les formulations faibles comme "ne sont pas explicitement mentionnees" si le contexte contient des indices exploitables. Prefere : "Le contexte permet d'identifier les axes suivants..."
- Cite les sources utiles avec le format [N], par exemple [1].
- Pour chaque solution ou capacite citee, indique au moins une source [N].
- Pour une source web, utilise aussi le format [N] ; ne recopie pas toute l'URL dans la reponse.
- Ne cite jamais seulement une page ou un chunk.
- N'utilise pas "source 1" en toutes lettres : utilise [1].
- Ne cite pas une source qui ne sert pas vraiment a la reponse.
- Quand plusieurs documents sont utiles, raisonne document par document puis fais une synthese.
- Si les documents se completent ou se contredisent, signale-le clairement.

Format prefere pour les questions techniques avec chiffres :
| Element | Valeur | Precision | Source |
| ... | ... | ... | ... |
{consignes_profil}
Question :
{question}

{aides_profil}

Contexte :
{contexte}
""".strip()

    return prompt

# Pourquoi ce prompt ?
# -> Il force le modele a rester proche des documents.
# -> Il reduit le risque d'hallucination.
# -> Il demande au modele de citer ses sources.
