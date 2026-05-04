# Sales Engineering demo

Ce document explique comment presenter le projet comme un actif de vente
interne ou comme support d'entretien Sales Specialist / Sales Engineering.

## Pitch court

Ce repo montre comment transformer un corpus documentaire PDF en assistant RAG
source, controlable et adaptable. L'objectif n'est pas de faire une application
SaaS complete, mais de demontrer la comprehension d'une architecture IA utile
en contexte client : ingestion documentaire, recherche vectorielle, reranking,
citations, controle qualite et adaptation au vocabulaire metier.

## Message de valeur

Pour une equipe Sales ou SE, le probleme n'est pas seulement de generer du texte.
Le vrai enjeu est de retrouver rapidement la bonne information dans des
documents techniques, de l'expliquer clairement, et de garder une trace des
sources.

Le projet adresse ce besoin avec :

- une interface Streamlit simple pour poser des questions ;
- des sources cliquables pour verifier les reponses ;
- une trace RAG pour expliquer comment la reponse a ete construite ;
- un profil generique par defaut, reutilisable avec d'autres corpus PDF ;
- des profils metier optionnels pour ajouter synonymes, consignes et
  reformulations sans modifier le moteur.

## Scenario de demo recommande

1. Montrer le corpus

   Expliquer que les PDF ne sont pas pousses dans GitHub pour des raisons de
   securite et de droits. Le repo contient le moteur, pas les documents.

2. Poser une question simple

   Exemple generique :

   ```text
   Quels sont les points clés des documents ?
   ```

   Objectif : montrer la reponse sourcee et les liens vers les documents.

3. Poser une question large

   Exemple :

   ```text
   Compare les solutions et leurs cas d'usage.
   ```

   Objectif : montrer le multi-doc reasoning, le `top_k` adaptatif et le
   contexte long.

4. Activer la trace RAG

   Montrer que le projet n'est pas une boite noire :

   - requetes de recherche ;
   - scores FAISS et BM25 ;
   - chunks retenus ;
   - modele final utilise ;
   - temps de retrieval et de generation.

5. Changer de profil ou de modele

   Montrer que la logique est adaptable :

   - `generic` pour tout corpus PDF ;
   - `everpure` comme exemple de profil metier ;
   - changement du modele de reponse pour comparer style et qualite.

## Points techniques a expliquer simplement

| Sujet | Explication courte |
| --- | --- |
| Embeddings | Transformer le texte en vecteurs pour chercher par sens, pas seulement par mots exacts. |
| FAISS | Base vectorielle locale, rapide et simple pour un POC. |
| BM25 | Ajoute un signal lexical utile pour les noms produits, acronymes et valeurs techniques. |
| Reranking | Reclasse les meilleurs candidats pour reduire le bruit. |
| Sources | Chaque reponse doit pouvoir etre verifiee dans les documents. |
| Profils | Les synonymes metier sont configurables, pas codes en dur dans le moteur. |

## Pourquoi c'est credible pour un role SE

Le projet montre la capacite a :

- traduire un besoin metier en architecture technique ;
- construire un POC demonstrable ;
- expliquer les limites sans sur-vendre ;
- securiser les secrets et exclure les donnees sensibles du repo ;
- separer le moteur generique des adaptations metier ;
- mesurer et comparer les choix de retrieval.

## Limites assumees

Le projet est volontairement un POC local :

- pas de gestion multi-utilisateur ;
- pas de backend API separe ;
- FAISS local plutot qu'une base vectorielle managee ;
- evaluation encore simple ;
- PDF et index reconstruits localement ;
- qualite dependante du corpus et de l'extraction PDF.

Ces limites sont utiles en discussion : elles ouvrent naturellement vers les
etapes d'industrialisation.

## Roadmap possible

1. Extraire `rag/prompting.py` pour isoler contexte et prompt.
2. Enrichir le jeu d'evaluation retrieval.
3. Ajouter une comparaison de cout / latence / qualite par modele.
4. Ajouter une option Qdrant ou Weaviate pour une version plus scalable.
5. Ajouter une API FastAPI pour separer backend et interface.

## Phrase de conclusion

Ce projet montre une approche pragmatique : partir d'un besoin documentaire
concret, construire un RAG verifiable, garder les sources visibles, puis rendre
le moteur suffisamment generique pour etre reutilise sur d'autres domaines.
