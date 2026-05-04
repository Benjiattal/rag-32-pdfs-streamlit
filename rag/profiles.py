"""
Profils de domaine optionnels.

Objectif :
garder le moteur RAG generique par defaut, tout en permettant d'activer des
optimisations metier pour un corpus particulier.

Exemple :
- `generic` : aucune hypothese sur le contenu des PDF.
- `everpure` : synonymes et consignes specifiques Everpure/Pure Storage.

Le coeur du RAG ne doit pas contenir de vocabulaire metier en dur. Ce module
charge ces informations depuis des fichiers JSON dans `rag/domain_profiles/`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path

from rag.config import DEFAULT_DOMAIN_PROFILE


PROFILES_DIR = Path(__file__).resolve().parent / "domain_profiles"


@dataclass(frozen=True)
class ExpansionRule:
    """Une regle declarative d'expansion de requete."""

    text: str
    any_terms: tuple[str, ...] = ()
    all_terms: tuple[str, ...] = ()

    def matches(self, question: str) -> bool:
        """Retourne True si la regle s'applique a la question."""
        question_minuscule = question.lower()

        if self.all_terms and not all(
            term.lower() in question_minuscule for term in self.all_terms
        ):
            return False

        if self.any_terms and not any(
            term.lower() in question_minuscule for term in self.any_terms
        ):
            return False

        return bool(self.all_terms or self.any_terms)


@dataclass(frozen=True)
class TechnicalFocusRule:
    """
    Regle optionnelle pour recentrer un resultat technique.

    Exemple pedagogique :
    si une question mentionne un produit tres precis et que le profil sait que
    les specs sont dans un fichier identifiable, on peut privilegier ce fichier.
    Pour le profil generique, il n'y a aucune regle de ce type.
    """

    any_terms: tuple[str, ...] = ()
    filename_contains: tuple[str, ...] = ()

    def matches(self, question: str) -> bool:
        """Retourne True si la regle s'applique a la question."""
        question_minuscule = question.lower()
        return bool(self.any_terms) and any(
            term.lower() in question_minuscule for term in self.any_terms
        )


@dataclass(frozen=True)
class DomainProfile:
    """Configuration optionnelle specialisee pour un domaine documentaire."""

    name: str
    display_name: str
    description: str = ""
    query_rewrite_context: str = (
        "Tu aides un moteur de recherche RAG sur des documents PDF fournis par l'utilisateur."
    )
    query_rewrite_extra_constraints: tuple[str, ...] = ()
    expansion_rules: tuple[ExpansionRule, ...] = ()
    technical_focus_rules: tuple[TechnicalFocusRule, ...] = ()
    prompt_extra_rules: tuple[str, ...] = ()
    prompt_extra_formats: tuple[str, ...] = ()
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def is_generic(self) -> bool:
        return self.name == "generic"

    def expand_question(self, question: str) -> str:
        """
        Applique les expansions deterministes du profil.

        Pour le profil generique, aucune expansion metier n'est ajoutee.
        """
        enrichissements = [
            rule.text
            for rule in self.expansion_rules
            if rule.matches(question)
        ]

        if not enrichissements:
            return question

        return (
            question
            + "\n\nTermes de recherche additionnels : "
            + " ".join(enrichissements)
        )

    def technical_focus_filename_terms(self, question: str) -> tuple[str, ...]:
        """
        Retourne les morceaux de noms de fichiers a privilegier pour une question.

        C'est volontairement declaratif : le moteur ne connait pas les noms de
        produits, il applique seulement les regles du profil actif.
        """
        termes: list[str] = []

        for rule in self.technical_focus_rules:
            if rule.matches(question):
                termes.extend(rule.filename_contains)

        return tuple(terme.lower() for terme in termes if terme)

    def query_rewrite_extra_text(self) -> str:
        if not self.query_rewrite_extra_constraints:
            return ""

        return "\n".join(
            f"- {contrainte}" for contrainte in self.query_rewrite_extra_constraints
        )

    def prompt_extra_text(self) -> str:
        blocs: list[str] = []

        if self.prompt_extra_rules:
            blocs.extend(f"- {regle}" for regle in self.prompt_extra_rules)

        if self.prompt_extra_formats:
            blocs.append("")
            blocs.extend(self.prompt_extra_formats)

        return "\n".join(blocs).strip()


GENERIC_PROFILE = DomainProfile(
    name="generic",
    display_name="Generique",
    description="Profil par defaut : aucune hypothese metier sur les PDF.",
)


def charger_profile_depuis_json(chemin: Path) -> DomainProfile:
    """Charge un profil de domaine depuis un fichier JSON."""
    donnees = json.loads(chemin.read_text(encoding="utf-8"))
    regles = tuple(
        ExpansionRule(
            text=regle["text"],
            any_terms=tuple(regle.get("any_terms", [])),
            all_terms=tuple(regle.get("all_terms", [])),
        )
        for regle in donnees.get("expansion_rules", [])
    )
    regles_focus = tuple(
        TechnicalFocusRule(
            any_terms=tuple(regle.get("any_terms", [])),
            filename_contains=tuple(regle.get("filename_contains", [])),
        )
        for regle in donnees.get("technical_focus_rules", [])
    )

    return DomainProfile(
        name=donnees["name"],
        display_name=donnees.get("display_name", donnees["name"]),
        description=donnees.get("description", ""),
        query_rewrite_context=donnees.get(
            "query_rewrite_context",
            GENERIC_PROFILE.query_rewrite_context,
        ),
        query_rewrite_extra_constraints=tuple(
            donnees.get("query_rewrite_extra_constraints", [])
        ),
        expansion_rules=regles,
        technical_focus_rules=regles_focus,
        prompt_extra_rules=tuple(donnees.get("prompt_extra_rules", [])),
        prompt_extra_formats=tuple(donnees.get("prompt_extra_formats", [])),
    )


def lister_profils_disponibles() -> list[str]:
    """Liste les profils activables."""
    noms = {"generic"}

    if PROFILES_DIR.exists():
        noms.update(chemin.stem for chemin in PROFILES_DIR.glob("*.json"))

    return sorted(noms)


def charger_profile(nom: str | None = None) -> DomainProfile:
    """Charge le profil demande, ou le profil generique si inconnu."""
    nom = (nom or DEFAULT_DOMAIN_PROFILE).strip().lower()

    if nom in {"", "generic", "generique"}:
        return GENERIC_PROFILE

    chemin = PROFILES_DIR / f"{nom}.json"

    if not chemin.exists():
        return GENERIC_PROFILE

    return charger_profile_depuis_json(chemin)


def profil_domaine_actif() -> DomainProfile:
    """Retourne le profil actif via la variable RAG_DOMAIN_PROFILE."""
    return charger_profile(os.getenv("RAG_DOMAIN_PROFILE", DEFAULT_DOMAIN_PROFILE))


def profil_est_actif(nom: str) -> bool:
    """Petit helper lisible pour les heuristiques optionnelles."""
    return profil_domaine_actif().name == nom
